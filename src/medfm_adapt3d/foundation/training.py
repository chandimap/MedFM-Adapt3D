"""Frozen-encoder Adaptation with train/validation Evidence and a Closed Test Barrier."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import torch
from monai.losses import DiceCELoss

from medfm_adapt3d.baseline.artifacts import (
    file_hash,
    fingerprint,
    implementation_fingerprint,
    read_json,
    read_sealed,
    seal,
    write_json,
)
from medfm_adapt3d.baseline.model import state_fingerprint
from medfm_adapt3d.baseline.protocol import LockedProtocol
from medfm_adapt3d.baseline.sampling import PatientBalancedSampler, augment_pair
from medfm_adapt3d.baseline.selection import (
    CheckpointSelector,
    RunAccess,
    require_optimization_adequacy,
)
from medfm_adapt3d.baseline.training import evaluate, optimization_diagnostics
from medfm_adapt3d.engineering.reproducibility import collect_runtime_provenance
from medfm_adapt3d.foundation.model import ARMS, FrozenRepresentation, build_frozen_model
from medfm_adapt3d.foundation.provenance import ModelRecord, claim_gate

ARTIFACTS = (
    "result.json",
    "provenance.json",
    "selected_head.pt",
    "checkpoint_lock.json",
    "optimization_gate.json",
    "training.jsonl",
)
ProbePhase = Literal["smoke", "calibration"]


def run_probe(
    protocol: LockedProtocol,
    record: ModelRecord,
    *,
    arm: str,
    k: int,
    head_seed: int,
    phase: ProbePhase,
    output_root: Path,
    checkpoint: Path | None = None,
    device: str = "cpu",
) -> Path:
    """Smoke and calibration only. Neither mode can read held-out images, masks, or QC."""
    claim_gate(record)
    if phase not in ("smoke", "calibration") or arm not in ARMS:
        raise ValueError("Only smoke/calibration frozen-representation experiments are enabled.")
    cfg = protocol.config
    expected_seed = cfg.seeds.smoke if phase == "smoke" else cfg.seeds.calibration
    if head_seed != expected_seed:
        raise ValueError("Use the prespecified, phase-specific engineering head seed.")
    support = protocol.support_ids(k)
    if phase == "calibration" and k not in protocol.calibration_budgets:
        raise ValueError("A shared calibration recipe uses the smallest/largest feasible K.")
    if (protocol.index.root / "test_release.json").exists():
        raise ValueError("Test was already released; further representation calibration is closed.")
    # Reconstructing both membership and index metadata before any training-target access.
    LockedProtocol(type(protocol.index)(protocol.index.path), cfg, protocol.path)
    target_device = torch.device(device)
    if target_device.type not in ("cpu", "cuda"):
        raise ValueError("Choose CPU or CUDA explicitly.")
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, initialization = build_frozen_model(
        record,
        arm=arm,
        head_seed=head_seed,
        checkpoint=checkpoint,
    )
    model.to(target_device)
    run_dir = output_root / arm / phase / f"k{k:02d}" / f"head{head_seed:03d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        return _execute(
            protocol,
            record,
            model,
            initialization,
            support,
            phase,
            k,
            head_seed,
            run_dir,
            target_device,
        )
    except Exception as error:
        write_json(
            run_dir / "FAILED.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "message": str(error),
            },
        )
        raise


def _execute(
    protocol: LockedProtocol,
    record: ModelRecord,
    model: FrozenRepresentation,
    initialization: dict[str, Any],
    support: tuple[str, ...],
    phase: ProbePhase,
    k: int,
    head_seed: int,
    run_dir: Path,
    device: torch.device,
) -> Path:
    start = time.monotonic()
    cfg = protocol.config
    budget = cfg.budget(phase)
    encoder_hash = initialization["encoder_initial_sha256"]
    provenance = {
        "schema_version": "1.0",
        "experiment": "frozen multiscale dense linear probe",
        "task": cfg.task,
        "phase": phase,
        "arm": initialization["arm"],
        "k_patients": k,
        "head_seed": head_seed,
        "model": initialization,
        "model_record": record.to_dict(),
        "claim_gate": claim_gate(record),
        "runtime": asdict(collect_runtime_provenance()),
        "device": str(device),
        "cpu_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "tf32": False,
        "config_sha256": fingerprint(cfg.to_dict()),
        "baseline_configuration": cfg.to_dict(),
        "implementation_sha256": implementation_fingerprint(),
        "candidate_index_sha256": protocol.index.sha256,
        "preprocessing_sha256": protocol.index.payload["preprocessing_sha256"],
        "partition_sha256": protocol.sha256,
        "support_patients": protocol.payload["supports"][str(k)]["patient_ids"],
        "support_reader_targets": len(support),
        "optimization": asdict(budget),
        "synthetic": protocol.index.payload["synthetic"],
        "clinical_performance_claimed": False,
    }
    write_json(run_dir / "provenance.json", seal(provenance))
    # Calibration access enforces a closed test barrier even on synthetic smoke runs.
    access = RunAccess(protocol, k, "calibration")
    sampler = PatientBalancedSampler([protocol.index.by_id[i] for i in support], cfg.seeds.loader)
    rng = torch.Generator().manual_seed(cfg.seeds.augmentation)
    optimizer = torch.optim.AdamW(
        model.head.parameters(),
        lr=budget.learning_rate,
        weight_decay=budget.weight_decay,
    )
    if {id(p) for g in optimizer.param_groups for p in g["params"]} != {
        id(p) for p in model.parameters() if p.requires_grad
    }:
        raise RuntimeError("Optimizer membership must equal exactly the trainable head.")
    loss_function = DiceCELoss(
        sigmoid=True,
        lambda_dice=budget.dice_weight,
        lambda_ce=budget.ce_weight,
        smooth_nr=1e-5,
        smooth_dr=1e-5,
    )
    selector = CheckpointSelector(budget)
    selected = None
    stopping_reason = "maximum_steps"
    with (run_dir / "training.jsonl").open("x", encoding="utf-8", newline="\n") as log:
        for step in range(1, budget.max_steps + 1):
            model.train()
            batch = [
                augment_pair(
                    *access.read("train", sampler.next_id()),
                    generator=rng,
                    probability=cfg.sampling.flip_probability,
                )
                for _ in range(budget.batch_size)
            ]
            image = torch.stack([pair[0] for pair in batch]).to(device)
            target = torch.stack([pair[1] for pair in batch]).float().to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(image), target)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite probe loss.")
            loss.backward()
            if any(p.grad is not None for p in model.encoder.parameters()):
                raise RuntimeError("An encoder parameter received a gradient.")
            torch.nn.utils.clip_grad_norm_(
                model.head.parameters(),
                budget.gradient_clip_norm,
                error_if_nonfinite=True,
            )
            optimizer.step()
            if not all(torch.isfinite(p).all() for p in model.head.parameters()):
                raise RuntimeError("Non-finite updated head parameters.")
            log.write(json.dumps({"step": step, "loss": float(loss.detach())}, allow_nan=False))
            log.write("\n")
            if step % budget.validation_interval == 0 or step == budget.max_steps:
                model.assert_frozen(encoder_hash)
                validation = evaluate(
                    model,
                    access,
                    "validation",
                    threshold=cfg.evaluation.threshold,
                    device=device,
                )
                if selector.observe(
                    step=step,
                    patient_macro_dice=validation["patient_macro"]["dice"],
                    source="validation",
                ):
                    selected = {
                        n: v.detach().cpu().clone() for n, v in model.head.state_dict().items()
                    }
                log.flush()
                if selector.should_stop(step):
                    stopping_reason = "validation_early_stopping"
                    break
    if selected is None:
        raise RuntimeError("No validation-selected head was produced.")
    checkpoint = run_dir / "selected_head.pt"
    with checkpoint.open("xb") as stream:
        torch.save(selected, stream)
    lock = {
        "checkpoint_sha256": file_hash(checkpoint),
        "selection_source": "validation",
        "selected_step": selector.best_step,
        "partition_sha256": protocol.sha256,
        "candidate_index_sha256": protocol.index.sha256,
        "config_sha256": provenance["config_sha256"],
        "implementation_sha256": provenance["implementation_sha256"],
        "encoder_sha256": encoder_hash,
        "checkpoint_scope": "task head only; upstream encoder tensors excluded",
    }
    write_json(run_dir / "checkpoint_lock.json", seal(lock))
    access.lock(run_dir / "checkpoint_lock.json", checkpoint)
    model.head.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True), strict=True
    )
    validation = evaluate(
        model,
        access,
        "validation",
        threshold=cfg.evaluation.threshold,
        device=device,
    )
    if abs(validation["patient_macro"]["dice"] - selector.best_value) > 1e-10:
        raise RuntimeError("Reloaded head failed to reproduce its selected validation Dice.")
    access.final_validation_completed()
    model.assert_frozen(encoder_hash)
    diagnostics = optimization_diagnostics(
        selector,
        actual_steps=step,
        stopping_reason=stopping_reason,
        validation=validation,
        patient_exposure=dict(sampler.patient_exposure),
        target_exposure=dict(sampler.target_exposure),
        candidate_sizes={r.candidate_id: math.prod(r.shape_cdhw) for r in protocol.index.records},
    )
    reason = None
    try:
        require_optimization_adequacy(diagnostics)
    except RuntimeError as error:
        reason = str(error)
    write_json(
        run_dir / "optimization_gate.json",
        seal(
            {
                "status": "passed" if reason is None else "blocked",
                "reason": reason,
                "diagnostics": diagnostics,
                "test_release_authorized": False,
            }
        ),
    )
    if access.read_counts["test"] != 0:
        raise RuntimeError("Test isolation failed.")
    if implementation_fingerprint() != provenance["implementation_sha256"]:
        raise RuntimeError("Implementation changed while the experiment was running.")
    LockedProtocol(type(protocol.index)(protocol.index.path), cfg, protocol.path)
    result = {
        "schema_version": "1.0",
        "status": "completed",
        "phase": phase,
        "arm": initialization["arm"],
        "k_patients": k,
        "head_seed": head_seed,
        "actual_optimization_steps": step,
        "selected_step": selector.best_step,
        "stopping_reason": stopping_reason,
        "validation_history": selector.history,
        "validation": validation,
        "test": None,
        "test_access": {"read_counts": access.read_counts, "events": access.events},
        "encoder_final_sha256": state_fingerprint(model.encoder),
        "encoder_unchanged": True,
        "head_final_sha256": state_fingerprint(model.head),
        "head_changed": state_fingerprint(model.head) != initialization["head_initial_sha256"],
        "patient_training_exposure": dict(sampler.patient_exposure),
        "reader_target_training_exposure": dict(sampler.target_exposure),
        "duration_seconds": time.monotonic() - start,
        "synthetic": provenance["synthetic"],
        "evidence_scope": "implementation and train/validation optimization only",
        "clean_generalization_claimed": False,
        "clinical_performance_claimed": False,
    }
    write_json(run_dir / "result.json", seal(result))
    write_json(run_dir / "COMPLETED.json", {name: file_hash(run_dir / name) for name in ARTIFACTS})
    return run_dir


def read_probe_run(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    completion = read_json(path / "COMPLETED.json")
    if set(completion) != set(ARTIFACTS) or any(
        file_hash(path / name) != digest for name, digest in completion.items()
    ):
        raise ValueError("Completed probe artifact changed or is incomplete.")
    result, provenance = read_sealed(path / "result.json"), read_sealed(path / "provenance.json")
    if result["test"] is not None or result["test_access"]["read_counts"]["test"] != 0:
        raise ValueError("Frozen probe evidence must have zero test access.")
    if result["encoder_final_sha256"] != provenance["model"]["encoder_initial_sha256"]:
        raise ValueError("Encoder changed in the recorded evidence.")
    return result, provenance


def compare_probes(pretrained: Path, random: Path, output: Path) -> dict[str, Any]:
    """Descriptive validation comparison; no hypothesis test or held-out performance claim."""
    a, pa = read_probe_run(pretrained)
    b, pb = read_probe_run(random)
    keys = (
        "phase",
        "k_patients",
        "head_seed",
        "model_record",
        "config_sha256",
        "implementation_sha256",
        "candidate_index_sha256",
        "preprocessing_sha256",
        "partition_sha256",
        "support_patients",
        "optimization",
        "synthetic",
    )
    if pa["arm"] != "suprem_frozen" or pb["arm"] != "random_frozen":
        raise ValueError("Comparison requires pretrained then architecture-matched random arms.")
    if any(pa[key] != pb[key] for key in keys) or any(
        pa["model"][key] != pb["model"][key]
        for key in ("head_initial_sha256", "encoder_specification", "parameter_budget")
    ):
        raise ValueError("Paired runs differ in cohort, support, recipe, head, or source.")
    if pa["model"]["checkpoint_loading"] is None or pb["model"]["checkpoint_loading"] is not None:
        raise ValueError("Paired arms do not have their declared checkpoint origins.")
    gates = {
        "suprem_frozen": read_sealed(pretrained / "optimization_gate.json")["status"],
        "random_frozen": read_sealed(random / "optimization_gate.json")["status"],
    }
    if any(status not in ("passed", "blocked") for status in gates.values()):
        raise ValueError("Unknown optimization-adequacy status in paired evidence.")
    adequate = all(status == "passed" for status in gates.values())
    result = {
        "schema_version": "1.0",
        "evidence_scope": "descriptive validation-only comparison",
        "synthetic": pa["synthetic"],
        "pretrained_result_sha256": file_hash(pretrained / "result.json"),
        "random_result_sha256": file_hash(random / "result.json"),
        "partition_sha256": pa["partition_sha256"],
        "k_patients": pa["k_patients"],
        "head_seed": pa["head_seed"],
        "matched_head_initialization": True,
        "matched_frozen_architecture": True,
        "optimization_adequacy": gates,
        "performance_comparison_interpretable": adequate,
        "validation_patient_macro_dice_difference": (
            a["validation"]["patient_macro"]["dice"] - b["validation"]["patient_macro"]["dice"]
        )
        if adequate
        else None,
        "test_label_reads": 0,
        "clean_generalization_claimed": False,
        "clinical_performance_claimed": False,
        "interpretation": (
            "Only adequate runs receive a descriptive difference. Validation selected the heads; "
            "the difference is not an unbiased test estimate or a generalization claim."
        ),
    }
    write_json(output, seal(result))
    return result
