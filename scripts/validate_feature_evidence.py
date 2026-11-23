"""Validate feature-study receipts and their cross-file scientific claims; optional live smoke."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from medfm_adapt3d.baseline.artifacts import (
    file_hash,
    implementation_fingerprint,
    read_json,
    read_sealed,
)
from medfm_adapt3d.baseline.config import Optimization
from medfm_adapt3d.baseline.selection import CheckpointSelector, require_optimization_adequacy
from medfm_adapt3d.foundation.feature_adapter import FEATURE_ARMS
from medfm_adapt3d.foundation.feature_experiment import paired_difference, summarize
from medfm_adapt3d.foundation.provenance import CHECKPOINT_SHA256


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_comparison(c: dict, *, extended: bool = False) -> None:
    require(set(c["arms"]) == set(FEATURE_ARMS), "Four original arms required.")
    require(
        c["clinical_performance_claimed"] is False and c["external_transfer"] == "NOT RUN",
        "Unsupported endpoint claim.",
    )
    require(c["overlap_status"] == "NOT_ESTABLISHED", "Unsupported source independence.")
    cfg = c["baseline_configuration"]
    require(cfg["evaluation"]["threshold"] == 0.5, "Primary threshold changed.")
    histories = []
    for name, a in c["arms"].items():
        b, f, m = a["parameter_budget"], a["feature_audit"], a["validation_counts"]
        adapted = name.endswith("_feature")
        groups = {"probe", "up", "down"} if adapted else {"probe"}
        trainable = 369 if adapted else 241
        require(
            b["encoder_parameters"] == b["frozen_parameters"] == 4399344
            and b["probe_parameters"] == 241
            and b["adapter_parameters"] == trainable - 241
            and b["trainable_parameters"] == trainable
            and b["total_parameters"] == trainable + 4399344
            and math.isclose(b["trainable_percentage"], 100 * trainable / b["total_parameters"]),
            "Trainable budget differs from declared architecture.",
        )
        if name.startswith("suprem"):
            checkpoint = a["checkpoint"]
            require(
                checkpoint["checkpoint_sha256"] == CHECKPOINT_SHA256
                and checkpoint["encoder_tensor_coverage"] == 1
                and checkpoint["checkpoint_size_bytes"] == 56500623,
                "Unverified pretrained encoder.",
            )
        else:
            require(a["checkpoint"] is None, "Random arm origin changed.")
        require(
            a["test_reads"] == 0 and a["encoder_initial_sha256"] == a["encoder_final_sha256"],
            "Held-out reads or changed encoder.",
        )
        require(
            f["selected_logits_sha256"] == f["reloaded_logits_sha256"],
            "Selected logit replay failed.",
        )
        require(
            set(f["selected_group_changed"]) == groups
            and all(f["selected_group_changed"].values())
            and f["adaptation_status"] == "PASS",
            "Missing evidence of adaptation.",
        )
        require(
            a["cpu_memory"]["peak_observed_rss_bytes"] > 0
            and a["cpu_memory"]["gpu_memory_bytes"] is None,
            "CPU resource provenance invalid.",
        )
        diagnostics = a["optimization_diagnostics"]
        val = a["validation_patient_macro"]
        require(
            diagnostics["all_validation_predictions_empty"]
            == (m["empty_predictions"] == m["reader_targets"])
            and diagnostics["zero_validation_overlap"] == (val["dice"] == 0),
            "Collapse classification changed.",
        )
        reason = None
        try:
            require_optimization_adequacy(diagnostics)
        except RuntimeError as error:
            reason = str(error)
        require(
            a["adequacy"] == ("PASSED" if reason is None else "BLOCKED")
            and a["adequacy_reason"] == reason,
            "Optimization gate contradicts its recorded diagnostics.",
        )
        trajectory = f["validation_trajectory"]
        selection = max(
            (v for v in trajectory if not v["selected_replay"]),
            key=lambda v: v["patient_macro_dice"],
        )
        require(
            selection["step"] == a["selected_step"] == trajectory[-1]["step"]
            and trajectory[-1]["selected_replay"]
            and val["dice"]
            == trajectory[-1]["patient_macro_dice"]
            == selection["patient_macro_dice"],
            "Selected validation score, earliest best step or replay differs.",
        )
        if c["schema_version"] == "1.1":
            require(
                set(f["selected_nonzero_gradient_groups"])
                == set(f["selected_nonzero_update_groups"])
                == groups,
                "Missing prefixes of selected gradient/update evidence.",
            )
            history = f["draw_history"]
            require(
                history
                and [x["step"] for x in history]
                == [v["step"] for v in trajectory if not v["selected_replay"]],
                "Support draw checkpoints differ from the evaluation grid.",
            )
            budget = Optimization(**c["effective_optimization"])
            selection_check = CheckpointSelector(budget)
            for point in trajectory[:-1]:
                selection_check.observe(
                    step=point["step"],
                    patient_macro_dice=point["patient_macro_dice"],
                    source="validation",
                )
            plateau = selection_check.should_stop(a["steps"])
            require(
                a["stopping_reason"]
                == ("validation_early_stopping" if plateau else "maximum_steps")
                and diagnostics["budget_exhausted_without_patience_plateau"]
                == (not plateau and selection_check.stale_validations < budget.patience)
                and diagnostics["unvisited_training_targets"]
                == c["support_reader_targets"] - history[-1]["unique_targets"]
                and all(row["draws"] == row["step"] * budget.batch_size for row in history),
                "Exposure or optimization plateau contradicts observed updates.",
            )
            histories.append(history)
        if extended:
            require(
                not c["synthetic"]
                and c["k_patients"] in (1, 5)
                and c["head_seed"] == 991
                and m["reader_targets"] == 4
                and m["patients"] == 2,
                "Clinical cohort changed.",
            )
            schedule = c["effective_optimization"]
            require(
                schedule["batch_size"] == 1
                and schedule["max_steps"] == schedule["min_steps"] == a["steps"] == 256
                and schedule["validation_interval"] == 32
                and schedule["patience"] == 3,
                "Follow-up schedule differs.",
            )
            require(
                [x["step"] for x in history] == list(range(32, 257, 32))
                and history[-1]["draws"] == 256
                and history[-1]["unique_targets"] == c["support_reader_targets"]
                and history[-1]["min_target_visits"] > 0,
                "Incomplete support coverage.",
            )
    for common in zip(*histories, strict=False):
        require(
            all(h == common[0] for h in common), "Actual ordered support draws differ between arms."
        )
    for source in ("suprem", "random"):
        control, adapter = (c["arms"][source + "_" + kind] for kind in ("frozen", "feature"))
        require(
            control["encoder_initial_sha256"] == adapter["encoder_initial_sha256"]
            and control["feature_audit"]["initial_logits_sha256"]
            == adapter["feature_audit"]["initial_logits_sha256"],
            "Initial functions differ within a source.",
        )
        require(
            c["selected_validation_dice_difference_feature_minus_frozen"][source]
            == paired_difference(control, adapter),
            "Unsupported comparative result.",
        )


def validate_archive(root: Path) -> None:
    base = root / "evidence/feature_adaptation"
    follow = base / "followup"
    old = read_json(base / "verification.json")
    require(
        set(old["experiments"]) == {"lidc_k1", "lidc_k5", "phantom_k1", "phantom_calibration_k1"},
        "Historical comparison set changed.",
    )
    for name, digest in old["experiments"].items():
        path = base / (name + ".json")
        require(file_hash(path) == digest, "Historical evidence changed.")
        validate_comparison(read_sealed(path))
    manifest = read_sealed(follow / "index.json")
    expected = {"design", "geometry", "overfit_suprem_frozen", "overfit_suprem_feature"}
    for k in (1, 5):
        expected.update(
            {f"ct_k{k}", f"readback_k{k}", f"threshold_short_k{k}", f"threshold_extended_k{k}"}
        )
    require(set(manifest["files"]) == expected, "Missing follow-up evidence.")
    for name, digest in manifest["files"].items():
        require(
            file_hash(follow / (name + ".json")) == digest,
            "Follow-up file SHA-256 mismatch: " + name,
        )
    geom = read_sealed(follow / "geometry.json")
    require(
        geom["read_counts"] == {"train": 1, "validation": 0, "test": 0}
        and geom["voxels"] == 65**3
        and geom["disagreeing_mask_voxels"] == 0
        and geom["normalized_ct_max_absolute_error"] < 1e-4,
        "Geometry replay differs from the reported spot check.",
    )
    for arm in ("suprem_frozen", "suprem_feature"):
        o = read_sealed(follow / f"overfit_{arm}.json")
        require(
            o["arm"] == arm
            and o["candidate_hash"] == geom["candidate_hash"]
            and o["read_counts"] == geom["read_counts"]
            and o["encoder_before_sha256"] == o["encoder_after_sha256"]
            and all(o["groups_changed"].values())
            and o["uncached_logits_bitwise_equal_at_steps"] == [0, 512]
            and [r["step"] for r in o["readouts"]] == [0, 12, 64, 128, 256, 512]
            and o["terminal_criterion"] == 0.80
            and o["terminal_criterion_met"] == (o["readouts"][-1]["dice"] >= 0.80)
            and o["script_sha256"] == file_hash(root / "scripts/diagnose_ct_memorization.py"),
            "Training-only diagnostic is inconsistent.",
        )
        require(
            o["partition_sha256"] == geom["partition_sha256"]
            and o["candidate_index_sha256"] == geom["candidate_index_sha256"]
            and o["implementation_sha256"] == manifest["measured_package_source_sha256"]
            and o["checkpoint_sha256"] == CHECKPOINT_SHA256,
            "Training diagnostic read a different cohort, model or code.",
        )
        for row in o["readouts"]:
            require(
                row["dice"]
                == 2 * row["overlap_voxels"] / (row["reference_voxels"] + row["predicted_voxels"]),
                "Single-target Dice differs from voxel counts.",
            )
    for k in (1, 5):
        path = follow / f"ct_k{k}.json"
        c = read_sealed(path)
        validate_comparison(c, extended=True)
        require(
            c["implementation_sha256"] == manifest["measured_package_source_sha256"]
            and c["partition_sha256"] == geom["partition_sha256"]
            and c["candidate_index_sha256"] == geom["candidate_index_sha256"]
            and c["support_reader_targets"] == (4 if k == 1 else 22)
            and c["k_patients"] == k,
            "Cohort or implementation differs between follow-up receipts.",
        )
        audit = read_json(follow / f"readback_k{k}.json")
        require(
            audit["input_comparison_sha256"] == file_hash(path)
            and audit["status"] == "PASS"
            and set(audit["arms"]) == set(FEATURE_ARMS)
            and all(
                a["validation_logits_bitwise_equal"]
                and a["test_array_reads"] == 0
                and a["patients"] == 2
                and a["validation_observations"] == 4
                and a["metrics_per_observation"] == 5
                for a in audit["arms"].values()
            ),
            "Independent functional and metric readback differs.",
        )
        for label, clinical in [("short", base / f"lidc_k{k}.json"), ("extended", path)]:
            s = read_sealed(follow / f"threshold_{label}_k{k}.json")
            require(
                s["comparison_sha256"] == file_hash(clinical)
                and s["thresholds"] == [0.1, 0.25, 0.5, 0.75]
                and s["primary_threshold"] == 0.5
                and set(s["arms"]) == {"suprem_frozen", "suprem_feature"},
                "Threshold readback differs from its source.",
            )
            for arm, data in s["arms"].items():
                require(
                    data["read_counts"]
                    == {"train": c["support_reader_targets"], "validation": 4, "test": 0}
                    and data["split"]["validation"]["thresholds"]["0.5"]["patient_macro_dice"]
                    == read_sealed(clinical)["arms"][arm]["validation_patient_macro"]["dice"],
                    "Primary threshold readback or isolation differs.",
                )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--live",
        type=Path,
        help="Fresh run folder containing the comparison and four private worker receipts",
    )
    args = p.parse_args()
    validate_archive(Path.cwd())
    if args.live:
        c = read_sealed(args.live / "comparison.json")
        paths = {
            a: args.live
            / "runs"
            / a
            / c["phase"]
            / f"k{c['k_patients']:02d}"
            / f"head{c['head_seed']:03d}"
            for a in FEATURE_ARMS
        }
        reconstructed = summarize(paths)
        require(
            reconstructed.pop("content_sha256") is not None and reconstructed == c,
            "Live aggregate differs from workers.",
        )
        require(
            c["implementation_sha256"] == implementation_fingerprint(),
            "Live code fingerprint differs from this checkout.",
        )
        validate_comparison(c)
    print("Feature evidence links, matched controls, adaptation and scope verified.")
