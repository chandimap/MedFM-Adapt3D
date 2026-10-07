"""Step-budgeted Patient-balanced Training; No Test Information Enters Optimization."""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from monai.losses.dice import DiceCELoss

from medfm_adapt3d.baseline.artifacts import (
    file_hash,
    fingerprint,
    implementation_fingerprint,
    read_sealed,
    seal,
    write_json,
)
from medfm_adapt3d.baseline.config import LIMITS, Phase
from medfm_adapt3d.baseline.metrics import aggregate_patients, segmentation_metrics
from medfm_adapt3d.baseline.model import AlignedSegResNet, build_random_model, state_fingerprint
from medfm_adapt3d.baseline.protocol import LockedProtocol
from medfm_adapt3d.baseline.sampling import PatientBalancedSampler, augment_pair
from medfm_adapt3d.baseline.selection import (
    CheckpointSelector,
    RunAccess,
    require_optimization_adequacy,
)
from medfm_adapt3d.engineering.reproducibility import collect_runtime_provenance


def _git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True, timeout=2
        )
        return bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


@torch.inference_mode()
def evaluate(
    model: AlignedSegResNet,
    access: RunAccess,
    partition: str,
    *,
    threshold: float,
    device: torch.device,
) -> dict[str, Any]:
    """Unaugmented reader targets, physical spacing from the audited candidate grid."""
    model.eval()
    observations = []
    for identifier in access.allowed[partition]:
        image, reference = access.read(partition, identifier)
        logits = model(image.unsqueeze(0).to(device))
        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite evaluation logits; run cannot produce valid evidence.")
        prediction = logits.sigmoid()[0, 0].cpu().numpy() >= threshold
        record = access._protocol.index.by_id[identifier]
        observations.append(
            segmentation_metrics(
                prediction,
                reference[0].numpy(),
                spacing_mm_xyz=record.spacing_mm_xyz,
                patient_id=record.patient_id,
                candidate_id=identifier,
            )
        )
    return aggregate_patients(observations)


def verify_frozen_recipe(path: Path, protocol: LockedProtocol) -> dict[str, Any]:
    frozen = read_sealed(path)
    checks = {
        "config_sha256": fingerprint(protocol.config.to_dict()),
        "partition_sha256": protocol.sha256,
        "candidate_index_sha256": protocol.index.sha256,
        "implementation_sha256": implementation_fingerprint(),
    }
    if any(frozen.get(key) != value for key, value in checks.items()):
        raise ValueError("Frozen recipe no longer matches configuration, cohort, or partitions.")
    if frozen.get("test_information_used") is not False or not frozen.get("calibration_evidence"):
        raise ValueError("Definitive testing requires train/validation-only calibration evidence.")
    return frozen


def release_test(protocol: LockedProtocol, frozen: dict[str, Any]) -> None:
    """Once testing starts, the same private index cannot authorize another recipe."""
    path = protocol.index.root / "test_release.json"
    expected = {
        "frozen_recipe_sha256": fingerprint(frozen),
        "candidate_index_sha256": protocol.index.sha256,
    }
    if path.exists():
        if read_sealed(path) != expected:
            raise ValueError("Test cohort was already released under a different frozen recipe.")
    else:
        write_json(path, seal(expected))


def optimization_diagnostics(
    selector: CheckpointSelector,
    *,
    actual_steps: int,
    stopping_reason: str,
    validation: dict[str, Any],
    patient_exposure: dict[str, int],
    target_exposure: dict[str, int],
    candidate_sizes: dict[str, int],
) -> dict[str, Any]:
    """Assessing adequacy from completed finite optimization and final validation only."""
    exposures = list(patient_exposure.values())
    return {
        "checkpoint_selection_complete": selector.best_step > 0,
        "final_validation_complete": validation["counts"]["reader_targets"] > 0,
        "minimum_steps_satisfied": actual_steps >= selector.budget.min_steps,
        # Non-finite loss, gradients, weights or logits abort the actual loop.
        "numerically_finite": True,
        "budget_exhausted_without_patience_plateau": (
            stopping_reason == "maximum_steps"
            and selector.stale_validations < selector.budget.patience
        ),
        "all_validation_predictions_empty": validation["counts"]["empty_predictions"]
        == validation["counts"]["reader_targets"],
        "zero_validation_overlap": validation["patient_macro"]["dice"] == 0,
        "all_validation_predictions_full": bool(validation["observations"])
        and all(
            row["predicted_voxels"] == candidate_sizes[row["candidate_id"]]
            for row in validation["observations"]
        ),
        "unvisited_training_patients": sum(n == 0 for n in exposures),
        "patient_exposure_balanced": bool(exposures) and max(exposures) - min(exposures) <= 1,
        "unvisited_training_targets": sum(n == 0 for n in target_exposure.values()),
    }


def run_baseline(
    protocol: LockedProtocol,
    *,
    k: int,
    model_seed: int,
    phase: Phase,
    output_root: Path,
    device: str = "cpu",
    frozen_recipe: Path | None = None,
) -> Path:
    """Running one immutable experiment. Calibration never reads held-out test arrays."""
    cfg = protocol.config
    cfg.budget(phase)  # Validate the phase before creating an output directory.
    support_ids = protocol.support_ids(k)  # Impossible K fails before any output/training.
    if phase == "definitive":
        if model_seed not in cfg.seeds.model or frozen_recipe is None:
            raise ValueError("Definitive runs require a prespecified model seed and frozen recipe.")
        if protocol.index.payload["synthetic"]:
            raise ValueError(
                "Synthetic fixtures cannot be presented as definitive LIDC experiments."
            )
        frozen = verify_frozen_recipe(frozen_recipe, protocol)
    else:
        expected_seed = cfg.seeds.calibration if phase == "calibration" else cfg.seeds.smoke
        if model_seed != expected_seed:
            raise ValueError("Engineering phase must use its separate prespecified model seed.")
        if phase == "calibration" and k not in protocol.calibration_budgets:
            raise ValueError(
                "Calibration uses the smallest/largest feasible K: "
                f"{protocol.calibration_budgets}. One recipe is shared across K."
            )
        if phase == "calibration" and (protocol.index.root / "test_release.json").exists():
            raise ValueError("Calibration is closed: definitive test evaluation already started.")
        frozen = None
    target_device = torch.device(device)
    if target_device.type not in ("cpu", "cuda"):
        raise ValueError("Baseline devices are CPU or CUDA.")
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; choose --device cpu explicitly.")
    # CUBLAS must be configured before constructing any CUDA tensors.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run_dir = output_root / phase / f"k{k:02d}" / f"seed{model_seed:03d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        return _execute(
            protocol,
            k,
            model_seed,
            phase,
            run_dir,
            target_device,
            frozen,
            support_ids,
            frozen_recipe,
        )
    except Exception as error:
        write_json(
            run_dir / "FAILED.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "message": str(error),
                "phase": phase,
            },
        )
        raise


def _execute(
    protocol: LockedProtocol,
    k: int,
    model_seed: int,
    phase: Phase,
    run_dir: Path,
    device: torch.device,
    frozen: dict[str, Any] | None,
    support_ids: tuple[str, ...],
    frozen_recipe: Path | None,
) -> Path:
    cfg = protocol.config
    budget = cfg.budget(phase)
    start = time.monotonic()
    model, initialization = build_random_model(cfg.model, model_seed)
    model.to(device)
    provenance = {
        "schema_version": "1.0",
        "task": cfg.task,
        "phase": phase,
        "k_patients": k,
        "runtime": asdict(collect_runtime_provenance()),
        "git_dirty": _git_dirty(),
        "device": str(device),
        "cpu_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "tf32": False,
        "model": initialization,
        "seeds": {**asdict(cfg.seeds), "model": model_seed},
        "config_sha256": fingerprint(cfg.to_dict()),
        "implementation_sha256": implementation_fingerprint(),
        "baseline_configuration": cfg.to_dict(),
        "optimization": asdict(budget),
        "candidate_index_sha256": protocol.index.sha256,
        "preprocessing_sha256": protocol.index.payload["preprocessing_sha256"],
        "partition_sha256": protocol.sha256,
        "patient_assignments": protocol.payload["patients"],
        "support_patients": protocol.payload["supports"][str(k)]["patient_ids"],
        "support_reader_targets": len(support_ids),
        "threshold": cfg.evaluation.threshold,
        "frozen_recipe": frozen,
        "synthetic": protocol.index.payload["synthetic"],
        "interpretation_limits": list(LIMITS),
        "clinical_performance_claimed": False,
    }
    write_json(run_dir / "provenance.json", seal(provenance))
    access = RunAccess(protocol, k, phase)
    sampler = PatientBalancedSampler(
        [protocol.index.by_id[i] for i in support_ids], cfg.seeds.loader
    )
    augmentation_rng = torch.Generator().manual_seed(cfg.seeds.augmentation)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=budget.learning_rate, weight_decay=budget.weight_decay
    )
    loss_function = DiceCELoss(
        sigmoid=True,
        lambda_dice=budget.dice_weight,
        lambda_ce=budget.ce_weight,
        smooth_nr=1e-5,
        smooth_dr=1e-5,
    )
    selector = CheckpointSelector(budget)
    selected_state = None
    stopping_reason = "maximum_steps"
    step = 0
    with (run_dir / "training.jsonl").open("x", encoding="utf-8", newline="\n") as log:
        for step in range(1, budget.max_steps + 1):
            model.train()
            batch = [
                augment_pair(
                    *access.read("train", sampler.next_id()),
                    generator=augmentation_rng,
                    probability=cfg.sampling.flip_probability,
                )
                for _ in range(budget.batch_size)
            ]
            image = torch.stack([pair[0] for pair in batch]).to(device)
            reference = torch.stack([pair[1] for pair in batch]).float().to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            loss = loss_function(logits, reference)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; baseline optimization failed.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), budget.gradient_clip_norm, error_if_nonfinite=True
            )
            optimizer.step()
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                raise RuntimeError("Non-finite updated model weights; held-out test stays closed.")
            log.write(
                json.dumps(
                    {
                        "step": step,
                        "training_loss": float(loss.detach()),
                        "learning_rate": budget.learning_rate,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            if step % budget.validation_interval == 0 or step == budget.max_steps:
                validation = evaluate(
                    model, access, "validation", threshold=cfg.evaluation.threshold, device=device
                )
                improved = selector.observe(
                    step=step,
                    patient_macro_dice=validation["patient_macro"]["dice"],
                    source="validation",
                )
                if improved:
                    selected_state = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }
                log.flush()
                if selector.should_stop(step):
                    stopping_reason = "validation_early_stopping"
                    break
    if selected_state is None:
        raise RuntimeError("No validation-selected checkpoint was produced.")
    checkpoint = run_dir / "selected_checkpoint.pt"
    with checkpoint.open("xb") as stream:
        torch.save(selected_state, stream)
    lock = {
        "schema_version": "1.0",
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": file_hash(checkpoint),
        "selection_source": "validation",
        "selection_metric": "patient_macro_dice",
        "selected_step": selector.best_step,
        "selected_validation_dice": selector.best_value,
        "candidate_index_sha256": protocol.index.sha256,
        "partition_sha256": protocol.sha256,
        "config_sha256": provenance["config_sha256"],
        "implementation_sha256": provenance["implementation_sha256"],
        "test_label_reads_before_lock": 0,
    }
    write_json(run_dir / "checkpoint_lock.json", seal(lock))
    # This is the run's own checkpoint, never an initialization/warm-start source.
    access.lock(run_dir / "checkpoint_lock.json", checkpoint)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True), strict=True
    )
    selected_hash = state_fingerprint(model)
    final_validation = evaluate(
        model, access, "validation", threshold=cfg.evaluation.threshold, device=device
    )
    if abs(final_validation["patient_macro"]["dice"] - selector.best_value) > 1e-10:
        raise RuntimeError("Locked checkpoint failed to reproduce its selected validation Dice.")
    access.final_validation_completed()
    diagnostics = optimization_diagnostics(
        selector,
        actual_steps=step,
        stopping_reason=stopping_reason,
        validation=final_validation,
        patient_exposure=dict(sampler.patient_exposure),
        target_exposure=dict(sampler.target_exposure),
        candidate_sizes={r.candidate_id: math.prod(r.shape_cdhw) for r in protocol.index.records},
    )
    # Preserve engineering failures for calibration review. Definitive failures
    # raise before release_test/open_test, never after viewing held-out scores.
    gate_error = None
    try:
        require_optimization_adequacy(diagnostics)
    except RuntimeError as error:
        gate_error = str(error)
    gate_path = run_dir / "optimization_gate.json"
    write_json(
        gate_path,
        seal(
            {
                "schema_version": "1.0",
                "status": "passed" if gate_error is None else "blocked",
                "reason": gate_error,
                "scope": "train-validation-only per-run release gate",
                "diagnostics": diagnostics,
                "actual_steps": step,
                "selected_step": selector.best_step,
                "checkpoint_sha256": lock["checkpoint_sha256"],
                "config_sha256": provenance["config_sha256"],
                "implementation_sha256": provenance["implementation_sha256"],
                "candidate_index_sha256": protocol.index.sha256,
                "partition_sha256": protocol.sha256,
                "test_label_reads": access.read_counts["test"],
            }
        ),
    )
    test = None
    # Real-data engineering must not expose the future definitive test cohort.
    # Synthetic smoke fixtures may exercise this barrier without clinical labels.
    if phase == "definitive" or (phase == "smoke" and protocol.index.payload["synthetic"]):
        if phase == "definitive":
            if frozen is None:
                raise RuntimeError("Definitive test release requires a frozen recipe.")
            access.approve_optimization(gate_path)
            # Recheck implementation/settings/receipt after training, before release.
            if frozen_recipe is None or verify_frozen_recipe(frozen_recipe, protocol) != frozen:
                raise ValueError("Frozen recipe/source drift blocks definitive test release.")
            release_test(protocol, frozen)
        access.open_test()
        test = evaluate(model, access, "test", threshold=cfg.evaluation.threshold, device=device)
        access.finish_test()
    if (
        state_fingerprint(model) != selected_hash
        or file_hash(checkpoint) != lock["checkpoint_sha256"]
    ):
        raise RuntimeError("Locked model changed during final evaluation.")
    result = {
        "schema_version": "1.0",
        "status": "completed",
        "phase": phase,
        "k_patients": k,
        "model_seed": model_seed,
        "config_sha256": provenance["config_sha256"],
        "partition_sha256": protocol.sha256,
        "candidate_index_sha256": protocol.index.sha256,
        "actual_optimization_steps": step,
        "selected_step": selector.best_step,
        "stopping_reason": stopping_reason,
        "validation_history": selector.history,
        "checkpoint": lock,
        "validation": final_validation,
        "test": test,
        "patient_training_exposure": dict(sampler.patient_exposure),
        "reader_target_training_exposure": dict(sampler.target_exposure),
        "test_access": {"events": access.events, "read_counts": access.read_counts},
        "optimization_diagnostics": diagnostics,
        "optimization_gate_sha256": file_hash(gate_path),
        "duration_seconds": time.monotonic() - start,
        "provenance_sha256": fingerprint(provenance),
        "selected_state_sha256": selected_hash,
        "synthetic": protocol.index.payload["synthetic"],
        "interpretation_limits": list(LIMITS),
    }
    write_json(run_dir / "result.json", seal(result))
    write_json(
        run_dir / "COMPLETED.json",
        {
            "result_sha256": file_hash(run_dir / "result.json"),
            "provenance_sha256": file_hash(run_dir / "provenance.json"),
            "checkpoint_sha256": lock["checkpoint_sha256"],
            "optimization_gate_sha256": file_hash(gate_path),
        },
    )
    return run_dir
