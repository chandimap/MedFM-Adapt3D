"""Reporting unit fixtures are synthetic scores, never model-performance evidence."""

from pathlib import Path

import pytest

from medfm_adapt3d.baseline.artifacts import (
    fingerprint,
    implementation_fingerprint,
    read_sealed,
    seal,
    write_json,
)
from medfm_adapt3d.baseline.protocol import LockedProtocol
from medfm_adapt3d.baseline.reporting import summarize_runs


def _report_fixture(protocol: LockedProtocol, root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    frozen = {
        "config_sha256": fingerprint(protocol.config.to_dict()),
        "partition_sha256": protocol.sha256,
        "candidate_index_sha256": protocol.index.sha256,
        "implementation_sha256": implementation_fingerprint(),
        "test_information_used": False,
        "calibration_evidence": [{"source": "synthetic reporting unit fixture"}],
    }
    path = root / "receipt.json"
    write_json(path, seal(frozen))
    for k in (1, 5, 10, 20):
        for seed in protocol.config.seeds.model:
            marker = (
                root / "runs" / "definitive" / f"k{k:02d}" / f"seed{seed:03d}" / "COMPLETED.json"
            )
            write_json(marker, {"synthetic_unit_fixture": True})

    def synthetic_completed_record(directory: Path):
        seed = int(directory.name.removeprefix("seed"))
        k = int(directory.parent.name.removeprefix("k"))
        score = (protocol.config.seeds.model.index(seed) + 1) / 10
        metrics = {
            "dice": score,
            "hd95_mm": 2.0,
            "absolute_volume_error_mm3": 1.0,
            "voxel_sensitivity": score,
            "empty_prediction_rate": 0.0,
        }
        partition = {
            "patient_macro": metrics,
            "counts": {
                "patients": 6,
                "reader_targets": 9,
                "empty_predictions": 0,
                "undefined_hd95_targets": 0,
                "fully_defined_hd95_patients": 6,
            },
        }
        return (
            {
                "phase": "definitive",
                "k_patients": k,
                "model_seed": seed,
                "synthetic": False,
                "config_sha256": frozen["config_sha256"],
                "partition_sha256": protocol.sha256,
                "candidate_index_sha256": protocol.index.sha256,
                "validation": partition,
                "test": partition,
            },
            {"frozen_recipe": frozen},
        )

    # Only storage decoding is stubbed for this reporting unit test. Separate
    # integration tests train real SegResNet and verify artifact decoding.
    monkeypatch.setattr(
        "medfm_adapt3d.baseline.reporting.read_completed_run", synthetic_completed_record
    )
    return path


def test_summary_retains_all_20_run_scores_and_emits_label_curve_table(
    locked_protocol: LockedProtocol,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _report_fixture(locked_protocol, tmp_path, monkeypatch)
    output = tmp_path / "summary"
    summarize_runs(
        locked_protocol, output_root=tmp_path / "runs", output=output, frozen_recipe=receipt
    )
    summary = read_sealed(output / "seed_summary.json")
    assert summary["population_inference"] is False
    for k in ("1", "5", "10", "20"):
        budget = summary["budgets"][k]
        assert budget["complete"]
        assert budget["completed_seeds"] == list(locked_protocol.config.seeds.model)
        assert budget["partitions"]["test"]["dice"]["values"] == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert budget["partitions"]["test"]["dice"]["mean"] == pytest.approx(0.3)
    assert len((output / "label_efficiency_seeds.csv").read_text().splitlines()) == 41


def test_missing_seed_fails_and_incomplete_summary_is_explicit(
    locked_protocol: LockedProtocol,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _report_fixture(locked_protocol, tmp_path, monkeypatch)
    (tmp_path / "runs/definitive/k01/seed017/COMPLETED.json").unlink()
    output = tmp_path / "summary"
    with pytest.raises(ValueError, match="missing seeds"):
        summarize_runs(
            locked_protocol, output_root=tmp_path / "runs", output=output, frozen_recipe=receipt
        )
    assert not output.exists()
    summarize_runs(
        locked_protocol,
        output_root=tmp_path / "runs",
        output=output,
        frozen_recipe=receipt,
        allow_incomplete=True,
    )
    first = read_sealed(output / "seed_summary.json")["budgets"]["1"]
    assert not first["complete"] and first["missing_seeds"] == [17]
