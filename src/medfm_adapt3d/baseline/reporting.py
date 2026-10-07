"""Calibration Receipts and Complete-seed Label-efficiency Tables."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from medfm_adapt3d.baseline.artifacts import (
    file_hash,
    fingerprint,
    implementation_fingerprint,
    read_json,
    read_sealed,
    seal,
    write_json,
)
from medfm_adapt3d.baseline.config import LIMITS, METRICS
from medfm_adapt3d.baseline.metrics import seed_statistics
from medfm_adapt3d.baseline.protocol import LockedProtocol
from medfm_adapt3d.baseline.selection import require_optimization_adequacy
from medfm_adapt3d.baseline.training import verify_frozen_recipe


def read_completed_run(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if (path / "FAILED.json").exists():
        raise ValueError("Failed runs cannot be evidence for calibration or results.")
    complete = read_json(path / "COMPLETED.json")
    for name in ("result", "provenance"):
        if complete[f"{name}_sha256"] != file_hash(path / f"{name}.json"):
            raise ValueError(f"Completed {name} artifact changed.")
    result, provenance = read_sealed(path / "result.json"), read_sealed(path / "provenance.json")
    checkpoint = path / "selected_checkpoint.pt"
    if complete["checkpoint_sha256"] != file_hash(checkpoint):
        raise ValueError("Completed checkpoint changed.")
    if complete["optimization_gate_sha256"] != file_hash(path / "optimization_gate.json"):
        raise ValueError("Completed optimization gate changed.")
    gate = read_sealed(path / "optimization_gate.json")
    if result["optimization_gate_sha256"] != file_hash(path / "optimization_gate.json") or (
        result["optimization_diagnostics"] != gate["diagnostics"]
    ):
        raise ValueError("Completed result disagrees with pre-test optimization evidence.")
    if result["phase"] == "definitive":
        require_optimization_adequacy(gate["diagnostics"])
        events = [event["event"] for event in result["test_access"]["events"]]
        if (
            events
            != [
                "checkpoint_locked",
                "final_validation_completed",
                "optimization_adequacy_passed",
                "test_evaluation_started",
                "test_evaluation_completed",
            ]
            or gate["test_label_reads"] != 0
        ):
            raise ValueError("Definitive evidence did not pass optimization before test.")
    if result["status"] != "completed" or result["provenance_sha256"] != fingerprint(provenance):
        raise ValueError("Completed result/provenance integrity mismatch.")
    required = (
        "schema_version",
        "runtime",
        "model",
        "seeds",
        "support_patients",
        "patient_assignments",
        "optimization",
        "preprocessing_sha256",
        "candidate_index_sha256",
        "config_sha256",
        "git_dirty",
        "phase",
        "interpretation_limits",
    )
    if any(key not in provenance for key in required):
        raise ValueError("Reproducibility provenance schema is incomplete.")
    if len(provenance["support_patients"]) != result["k_patients"]:
        raise ValueError("Completed result K does not count support patients.")
    for key in ("config_sha256", "partition_sha256", "candidate_index_sha256", "phase"):
        if result[key] != provenance[key]:
            raise ValueError("Result disagrees with its locked provenance.")
    return result, provenance


def freeze_recipe(
    protocol: LockedProtocol,
    calibration_runs: list[Path],
    *,
    output: Path,
    rationale: str,
) -> None:
    """Requiring inspectable, train/validation-only optimization evidence before test.

    At most three recipe trials on the smallest/largest feasible K. The
    selected recipe must show a patience plateau and non-collapsed validation
    predictions. This is an engineering gate, not proof of global optimality.
    """
    if not rationale.strip() or not calibration_runs:
        raise ValueError("A train/validation rationale and calibration run artifacts are required.")
    if (protocol.index.root / "test_release.json").exists():
        raise ValueError("Cannot recalibrate/refreeze a cohort after definitive test release.")
    if len(set(p.resolve() for p in calibration_runs)) != len(calibration_runs):
        raise ValueError("Duplicated calibration evidence is not a repeated experiment.")
    if len(calibration_runs) > 3 * len(protocol.calibration_budgets):
        raise ValueError("Limited calibration allows at most three recipes at feasible extremes.")
    evidence = []
    selected_budgets = set()
    recipe_hashes = set()
    current_hash = fingerprint(protocol.config.to_dict())
    for path in calibration_runs:
        result, provenance = read_completed_run(path)
        if result["phase"] != "calibration" or result["test"] is not None:
            raise ValueError("Calibration evidence must never contain test evaluation.")
        if result["test_access"]["read_counts"]["test"] != 0:
            raise ValueError("Test label access invalidates calibration evidence.")
        if result["partition_sha256"] != protocol.sha256 or (
            result["candidate_index_sha256"] != protocol.index.sha256
        ):
            raise ValueError("Calibration must use the same fixed cohort and support protocol.")
        if result["k_patients"] not in protocol.calibration_budgets:
            raise ValueError("Calibration requires the smallest/largest feasible patient budgets.")
        if result["model_seed"] != protocol.config.seeds.calibration:
            raise ValueError("Calibration seed must be separate and prespecified.")
        recipe_hashes.add(result["config_sha256"])
        if result["config_sha256"] == current_hash:
            if provenance["implementation_sha256"] != implementation_fingerprint():
                raise ValueError(
                    "Calibration evidence belongs to a different package implementation."
                )
            if result["actual_optimization_steps"] < protocol.config.optimization.min_steps:
                raise ValueError("Calibration did not receive the minimum optimization budget.")
            try:
                require_optimization_adequacy(result["optimization_diagnostics"])
            except RuntimeError as error:
                raise ValueError(
                    f"Calibration cannot justify the shared recipe: {error}"
                ) from error
            selected_budgets.add(result["k_patients"])
        evidence.append(
            {
                "phase": "calibration",
                "k": result["k_patients"],
                "config_sha256": result["config_sha256"],
                "result_sha256": file_hash(path / "result.json"),
                "provenance_sha256": fingerprint(provenance),
                "validation_dice": result["validation"]["patient_macro"]["dice"],
                "actual_steps": result["actual_optimization_steps"],
            }
        )
    required = set(protocol.calibration_budgets)
    if len(recipe_hashes) > 3 or selected_budgets != required:
        raise ValueError("Freeze requires ONE current recipe at both smallest/largest feasible K.")
    write_json(
        output,
        seal(
            {
                "schema_version": "1.0",
                "config_sha256": current_hash,
                "candidate_index_sha256": protocol.index.sha256,
                "partition_sha256": protocol.sha256,
                "test_information_used": False,
                "implementation_sha256": implementation_fingerprint(),
                "rationale": rationale,
                "calibration_evidence": evidence,
                "calibration_budgets": list(protocol.calibration_budgets),
                "interpretation_limits": list(LIMITS),
            }
        ),
    )


def summarize_runs(
    protocol: LockedProtocol,
    *,
    output_root: Path,
    output: Path,
    frozen_recipe: Path,
    allow_incomplete: bool = False,
) -> None:
    """Failing on missing seeds by default; keeping every seed and both partition metrics.

    The CSV directly supports future label-efficiency plots. No best-seed filter,
    patient confidence interval, clinical claim, or causal-pretraining claim.
    """
    frozen = verify_frozen_recipe(frozen_recipe, protocol)
    rows, summaries = [], {}
    for k_text in sorted(protocol.payload["supports"], key=int):
        k = int(k_text)
        results, missing = [], []
        for seed in protocol.config.seeds.model:
            path = output_root / "definitive" / f"k{k:02d}" / f"seed{seed:03d}"
            if not (path / "COMPLETED.json").exists():
                missing.append(seed)
                continue
            result, provenance = read_completed_run(path)
            if (
                result["phase"],
                result["k_patients"],
                result["model_seed"],
                result["synthetic"],
            ) != (
                "definitive",
                k,
                seed,
                False,
            ) or result["test"] is None:
                raise ValueError("Summary contains a non-definitive or mismatched run.")
            if (
                any(
                    result[key] != frozen[key]
                    for key in (
                        "config_sha256",
                        "candidate_index_sha256",
                        "partition_sha256",
                    )
                )
                or provenance["frozen_recipe"] != frozen
            ):
                raise ValueError("Summary contains mixed cohorts or optimization recipes.")
            results.append(result)
            for partition in ("validation", "test"):
                rows.append(
                    {
                        "k_patients": k,
                        "model_seed": seed,
                        "partition": partition,
                        **result[partition]["patient_macro"],
                        **result[partition]["counts"],
                    }
                )
        if missing and not allow_incomplete:
            raise ValueError(
                f"K={k} is missing seeds {missing}; all prespecified seeds are required."
            )
        summaries[k_text] = {
            "completed_seeds": [r["model_seed"] for r in results],
            "missing_seeds": missing,
            "complete": not missing,
            "partitions": {
                partition: {
                    metric: seed_statistics(
                        [r[partition]["patient_macro"][metric] for r in results]
                    )
                    for metric in (*METRICS, "empty_prediction_rate")
                }
                for partition in ("validation", "test")
            },
        }
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "seed_summary.json",
        seal(
            {
                "schema_version": "1.0",
                "budgets": summaries,
                "frozen_recipe": frozen,
                "unavailable_budgets": protocol.payload["unavailable_budgets"],
                "interpretation_limits": list(LIMITS),
                "seed_sd_ddof": 1,
                "quartiles": "linear 25th/75th percentiles",
                "population_inference": False,
            }
        ),
    )
    if rows:
        with (output / "label_efficiency_seeds.csv").open(
            "x", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
