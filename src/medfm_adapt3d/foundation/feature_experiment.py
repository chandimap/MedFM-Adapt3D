"""Run and verify a matched train/validation feature-adaptation experiment on CPU."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from medfm_adapt3d.baseline.artifacts import file_hash, fingerprint, read_sealed, seal, write_json
from medfm_adapt3d.baseline.candidates import CandidateIndex
from medfm_adapt3d.baseline.config import SmokeBudget, load_baseline_config
from medfm_adapt3d.baseline.model import state_fingerprint
from medfm_adapt3d.baseline.protocol import LockedProtocol, create_partition
from medfm_adapt3d.baseline.selection import require_optimization_adequacy
from medfm_adapt3d.baseline.synthetic import create_synthetic_candidates
from medfm_adapt3d.foundation.feature_adapter import (
    FEATURE_ARMS,
    FeatureRecipe,
    configure_feature_arm,
    trainable_groups,
)
from medfm_adapt3d.foundation.feature_audit import output_fingerprint
from medfm_adapt3d.foundation.model import FrozenRepresentation, build_frozen_model
from medfm_adapt3d.foundation.provenance import (
    CHECKPOINT_BYTES,
    CHECKPOINT_SHA256,
    ModelRecord,
    load_model_record,
)
from medfm_adapt3d.foundation.training import read_probe_run, run_probe

MATCH_FIELDS = (
    "task",
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
    "support_reader_targets",
    "optimization",
    "device",
    "cpu_threads",
    "runtime",
    "deterministic_algorithms",
    "synthetic",
)


def _arm_summary(path: Path, arm: str) -> tuple[dict[str, Any], dict[str, Any]]:
    result, p = read_probe_run(path)
    model = p["model"]
    audit = result["feature_audit"]
    lock = read_sealed(path / "checkpoint_lock.json")
    gate = read_sealed(path / "optimization_gate.json")
    loading = model["checkpoint_loading"]
    if arm != result["arm"] or arm != p["arm"] or arm != model["arm"] or arm != lock["arm"]:
        raise ValueError("Arm identity mismatch.")
    if arm.startswith("suprem"):
        if not loading or (
            loading["checkpoint_sha256"],
            loading["checkpoint_size_bytes"],
            loading["encoder_tensor_coverage"],
        ) != (CHECKPOINT_SHA256, CHECKPOINT_BYTES, 1):
            raise ValueError("Missing verified official checkpoint identity/coverage.")
    elif loading is not None:
        raise ValueError("Random arm unexpectedly loaded a checkpoint.")
    if (
        lock["checkpoint_sha256"] != file_hash(path / "selected_task.pt")
        or lock["encoder_sha256"] != result["encoder_final_sha256"]
        or lock["feature_recipe_sha256"] != fingerprint(model["feature_recipe"])
        or lock["selected_logits_sha256"] != audit["reloaded_logits_sha256"]
        or lock["selected_step"] != result["selected_step"]
    ):
        raise ValueError("Selected state, recipe or replay does not match its checkpoint lock.")
    if lock["selection_source"] != "validation" or any(
        lock[key] != p[key]
        for key in (
            "partition_sha256",
            "config_sha256",
            "implementation_sha256",
            "candidate_index_sha256",
        )
    ):
        raise ValueError("Checkpoint lock differs from the training provenance.")
    reason = None
    try:
        require_optimization_adequacy(gate["diagnostics"])
    except RuntimeError as error:
        reason = str(error)
    if (
        gate["status"] != ("passed" if reason is None else "blocked")
        or gate["reason"] != reason
        or gate["test_release_authorized"] is not False
    ):
        raise ValueError("Invalid optimization gate or reason.")
    with (path / "training.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if [r["step"] for r in rows] != list(range(1, result["actual_optimization_steps"] + 1)):
        raise ValueError("Training trajectory is incomplete.")
    validation = result["validation"]
    history = result["validation_history"]
    best = max(history, key=lambda row: row["patient_macro_dice"])
    if (
        best["step"] != lock["selected_step"]
        or best["source"] != "validation"
        or best["patient_macro_dice"] != validation["patient_macro"]["dice"]
    ):
        raise ValueError("Validation trajectory conflicts with the selected checkpoint.")
    if "selected_nonzero_gradient_groups" in audit:
        prefix = rows[: best["step"]]
        groups = set(audit["initial_group_sha256"])

        def active(field: str) -> set[str]:
            return {g for r in prefix for g in groups if r[field][g] > 0}

        gradients, updates = active("gradient_l2_before_clip"), active("parameter_update_l2")
        changed = {
            g
            for g in groups
            if audit["initial_group_sha256"][g] != audit["selected_group_sha256"][g]
        }
        expected_status = "PASS" if groups == gradients == updates == changed else "FAIL"
        if (
            gradients != set(audit["selected_nonzero_gradient_groups"])
            or updates != set(audit["selected_nonzero_update_groups"])
            or audit["adaptation_status"] != expected_status
        ):
            raise ValueError("Selected gradient/update claims conflict with training evidence.")
    summary = {
        "checkpoint": None
        if loading is None
        else {
            k: loading[k]
            for k in (
                "checkpoint_sha256",
                "checkpoint_size_bytes",
                "encoder_tensor_count",
                "encoder_tensor_coverage",
                "verified_load_seconds",
            )
        },
        "encoder_initial_sha256": model["encoder_initial_sha256"],
        "encoder_final_sha256": result["encoder_final_sha256"],
        "head_initial_sha256": model["head_initial_sha256"],
        "parameter_budget": model["parameter_budget"],
        "optimizer_parameters": result["optimizer_parameters"],
        "steps": result["actual_optimization_steps"],
        "duration_seconds": result["duration_seconds"],
        "selected_step": result["selected_step"],
        "stopping_reason": result["stopping_reason"],
        "adequacy": gate["status"].upper(),
        "adequacy_reason": gate["reason"],
        "optimization_diagnostics": gate["diagnostics"],
        "validation_patient_macro": validation["patient_macro"],
        "validation_counts": validation["counts"],
        "feature_audit": audit,
        "cpu_memory": result["cpu_memory"],
        "training_first": rows[0],
        "training_last": rows[-1],
        "test_reads": result["test_access"]["read_counts"]["test"],
        "artifact_sha256": {
            n: file_hash(path / n)
            for n in (
                "training.jsonl",
                "result.json",
                "selected_task.pt",
                "checkpoint_lock.json",
                "provenance.json",
                "optimization_gate.json",
                "COMPLETED.json",
            )
        },
    }
    return summary, p


def restore_feature_model(
    run: Path, record: ModelRecord, *, checkpoint: Path | None
) -> FrozenRepresentation:
    """Rebuild the archived CPU function from its identified encoder and task-only state."""
    result, p = read_probe_run(run)
    arm = p["arm"]
    if arm not in FEATURE_ARMS or record.sha256 != fingerprint(p["model_record"]):
        raise ValueError("Restoration requires the recorded model release and experiment arm.")
    _arm_summary(run, arm)
    if torch.get_num_threads() != p["cpu_threads"]:
        raise ValueError("Set the recorded CPU thread count before exact logit replay.")
    model, init = build_frozen_model(
        record,
        arm=arm.replace("_feature", "_frozen"),
        head_seed=p["head_seed"],
        checkpoint=checkpoint,
    )
    recipe = FeatureRecipe(**p["model"]["feature_recipe"])
    configure_feature_arm(model, init, arm, recipe)
    audit = result["feature_audit"]
    if init["encoder_initial_sha256"] != result["encoder_final_sha256"] or (
        init["trainable_initial_sha256"] != audit["initial_group_sha256"]
    ):
        raise ValueError("Reconstructed initial weights do not match the archived experiment.")
    state = torch.load(run / "selected_task.pt", weights_only=True, map_location="cpu")
    model.head.load_state_dict(state, strict=True)
    model.eval()
    actual = {name: state_fingerprint(m) for name, m in trainable_groups(model).items()}
    if (
        actual != audit["selected_group_sha256"]
        or output_fingerprint(model) != audit["selected_logits_sha256"]
    ):
        raise ValueError("Fresh CPU reconstruction failed selected-state replay.")
    model.assert_frozen(init["encoder_initial_sha256"])
    return model


def paired_difference(control: dict[str, Any], intervention: dict[str, Any]) -> float | None:
    """Positive means larger selected validation Dice, conditional on adequate optimization."""
    if any(
        arm["adequacy"] != "PASSED" or arm["feature_audit"]["adaptation_status"] != "PASS"
        for arm in (control, intervention)
    ):
        return None
    return (
        intervention["validation_patient_macro"]["dice"]
        - control["validation_patient_macro"]["dice"]
    )


def summarize(paths: dict[str, Path]) -> dict[str, Any]:
    if set(paths) != set(FEATURE_ARMS):
        raise ValueError("All four prespecified arms are required.")
    arms, provenance = {}, {}
    for name in FEATURE_ARMS:
        arms[name], provenance[name] = _arm_summary(paths[name], name)
    base = provenance["suprem_frozen"]
    for p in provenance.values():
        if any(p[k] != base[k] for k in MATCH_FIELDS) or any(
            p["model"][k] != base["model"][k]
            for k in (
                "head_initial_sha256",
                "encoder_specification",
                "feature_recipe",
                "input_policy",
            )
        ):
            raise ValueError(
                "Cohort, recipe, hardware settings, source or head initialization differs."
            )
    differences = {}
    for source in ("suprem", "random"):
        a, b = (arms[f"{source}_{kind}"] for kind in ("frozen", "feature"))
        if (
            a["encoder_initial_sha256"] != b["encoder_initial_sha256"]
            or a["feature_audit"]["initial_logits_sha256"]
            != b["feature_audit"]["initial_logits_sha256"]
        ):
            raise ValueError("Within-source comparison did not start at the same function.")
        differences[source] = paired_difference(a, b)
    for kind in ("frozen", "feature"):
        a, b = (arms[f"{source}_{kind}"] for source in ("suprem", "random"))
        if a["parameter_budget"] != b["parameter_budget"] or (
            a["feature_audit"]["initial_group_sha256"] != b["feature_audit"]["initial_group_sha256"]
        ):
            raise ValueError(
                "Representation-source controls differ in trainable state or capacity."
            )
    histories = [a["feature_audit"].get("draw_history") for a in arms.values()]
    if any(h is not None for h in histories):
        if not all(histories):
            raise ValueError("Missing ordered support exposure evidence in a matched arm.")
        for prefix in zip(*histories, strict=False):
            if any(row != prefix[0] for row in prefix[1:]):
                raise ValueError("Actual ordered reader-target exposure differs between arms.")
    return seal(
        {
            "schema_version": "1.1",
            "effective_optimization": base["optimization"],
            "study": "paired CPU feature-output adaptation",
            "synthetic": base["synthetic"],
            "phase": base["phase"],
            "k_patients": base["k_patients"],
            "head_seed": base["head_seed"],
            "feature_recipe": base["model"]["feature_recipe"],
            "baseline_configuration": base["baseline_configuration"],
            "cpu_threads": base["cpu_threads"],
            "environment": base["runtime"],
            "official_model_id": base["model_record"]["model_id"],
            "official_checkpoint_revision": base["model_record"]["checkpoint_revision"],
            "overlap_status": base["model_record"]["overlap_status"],
            "implementation_sha256": base["implementation_sha256"],
            "candidate_index_sha256": base["candidate_index_sha256"],
            "partition_sha256": base["partition_sha256"],
            "preprocessing_sha256": base["preprocessing_sha256"],
            "support_reader_targets": base["support_reader_targets"],
            "arms": arms,
            "selected_validation_dice_difference_feature_minus_frozen": differences,
            "clinical_performance_claimed": False,
            "external_transfer": "NOT RUN",
            "interpretation": "Descriptive differences use selection validation only.",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--partitions", type=Path)
    parser.add_argument("--phase", choices=("smoke", "calibration"), default="smoke")
    parser.add_argument(
        "--ct-exposure",
        action="store_true",
        help="Fixed 256-step real-CT follow-up; batch one, validate every 32 steps.",
    )
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--recipe", type=Path, default=Path("configs/foundation/feature_adapter.yaml")
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/baseline/random_segresnet.yaml")
    )
    parser.add_argument(
        "--model-record", type=Path, default=Path("configs/foundation/suprem_segresnet.yaml")
    )
    parser.add_argument("--worker-arm", choices=FEATURE_ARMS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.threads < 1 or (args.index is None) != (args.partitions is None):
        parser.error("Provide positive threads and both private index and partitions together.")
    if args.index is None and (args.phase != "smoke" or args.k != 1 or args.worker_arm):
        parser.error("The built-in phantom experiment uses smoke, K=1.")
    torch.set_num_threads(args.threads)
    cfg, recipe = load_baseline_config(args.config), FeatureRecipe.load(args.recipe)
    if args.ct_exposure:
        if (
            args.index is None
            or args.phase != "smoke"
            or CandidateIndex(args.index).payload["synthetic"]
        ):
            parser.error("The exposure follow-up requires a real private index and smoke seed.")
        cfg = replace(
            cfg, smoke=SmokeBudget(max_steps=256, min_steps=256, validation_interval=32, patience=3)
        )
    record = load_model_record(args.model_record)
    seed = cfg.seeds.smoke if args.phase == "smoke" else cfg.seeds.calibration
    if args.worker_arm:
        protocol = LockedProtocol(CandidateIndex(args.index), cfg, args.partitions)
        run_probe(
            protocol,
            record,
            arm=args.worker_arm,
            k=args.k,
            head_seed=seed,
            phase=args.phase,
            output_root=args.output / "runs",
            feature_recipe=recipe,
            checkpoint=args.checkpoint if args.worker_arm.startswith("suprem") else None,
        )
        return
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        if (
            args.checkpoint.stat().st_size != CHECKPOINT_BYTES
            or file_hash(args.checkpoint) != CHECKPOINT_SHA256
        ):
            raise ValueError("Official checkpoint size or SHA-256 mismatch.")
        if args.index is None:
            args.index = create_synthetic_candidates(args.output / "candidates", patients=10)
            args.partitions = args.output / "partitions.json"
            create_partition(CandidateIndex(args.index), cfg, args.partitions)
        protocol = LockedProtocol(CandidateIndex(args.index), cfg, args.partitions)
        protocol.support_ids(args.k)
        paths = {}
        for arm in FEATURE_ARMS:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "medfm_adapt3d.foundation.feature_experiment",
                    "--worker-arm",
                    arm,
                    "--checkpoint",
                    str(args.checkpoint),
                    "--output",
                    str(args.output),
                    "--index",
                    str(args.index),
                    "--partitions",
                    str(args.partitions),
                    "--phase",
                    args.phase,
                    "--k",
                    str(args.k),
                    "--threads",
                    str(args.threads),
                    "--recipe",
                    str(args.recipe),
                    "--config",
                    str(args.config),
                    "--model-record",
                    str(args.model_record),
                    *(["--ct-exposure"] if args.ct_exposure else []),
                ],
                check=True,
            )
            paths[arm] = (
                args.output / "runs" / arm / args.phase / f"k{args.k:02d}" / f"head{seed:03d}"
            )
        write_json(args.output / "comparison.json", summarize(paths))
    except Exception as error:
        write_json(
            args.output / "FAILED.json",
            {"status": "FAIL", "error_type": type(error).__name__, "message": str(error)},
        )
        raise
    print(f"Completed four-arm train/validation experiment: {args.output / 'comparison.json'}")


if __name__ == "__main__":
    main()
