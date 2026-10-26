import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from medfm_adapt3d.baseline.artifacts import read_sealed, seal, write_json
from medfm_adapt3d.baseline.candidates import CandidateIndex
from medfm_adapt3d.baseline.config import BaselineConfig, SmokeBudget
from medfm_adapt3d.baseline.protocol import LockedProtocol, create_partition
from medfm_adapt3d.foundation import training
from medfm_adapt3d.foundation.provenance import load_model_record
from medfm_adapt3d.foundation.training import compare_probes, read_probe_run, run_probe


def _record():
    return load_model_record(Path("configs/foundation/suprem_segresnet.yaml"))


def _run(protocol: LockedProtocol, output: Path, **changes: object) -> Path:
    arguments = {
        "arm": "random_frozen",
        "k": 1,
        "head_seed": protocol.config.seeds.smoke,
        "phase": "smoke",
        "output_root": output,
    }
    arguments.update(changes)
    return run_probe(protocol, _record(), **arguments)


def test_training_is_reproducible_test_closed_and_head_only(
    locked_protocol: LockedProtocol,
    tmp_path: Path,
) -> None:
    config = replace(locked_protocol.config, smoke=SmokeBudget(2, 1, 1, 1))
    protocol = LockedProtocol(locked_protocol.index, config, locked_protocol.path)
    first, second = _run(protocol, tmp_path / "a"), _run(protocol, tmp_path / "b")
    a, pa = read_probe_run(first)
    b, _ = read_probe_run(second)
    assert a["head_final_sha256"] == b["head_final_sha256"]
    assert a["validation"] == b["validation"]
    assert a["head_changed"] and a["encoder_unchanged"]
    assert a["test"] is None and a["test_access"]["read_counts"]["test"] == 0
    assert pa["model"]["parameter_budget"]["trainable_parameters"] == 241
    assert (
        len(list(first.glob("*.pt"))) == 1 and (first / "selected_head.pt").stat().st_size < 10000
    )
    with pytest.raises(FileExistsError):
        _run(protocol, tmp_path / "a")
    with pytest.raises(ValueError, match="pretrained then"):
        compare_probes(first, second, tmp_path / "wrong-comparison.json")


@pytest.mark.parametrize(
    "changes",
    [
        {"phase": "definitive"},
        {"head_seed": 17},
        {"arm": "anything"},
        {"k": 999},
    ],
)
def test_unsupported_runs_fail_before_creating_outputs(
    locked_protocol: LockedProtocol,
    tmp_path: Path,
    changes: dict[str, object],
) -> None:
    output = tmp_path / "forbidden"
    with pytest.raises(ValueError):
        _run(locked_protocol, output, **changes)
    assert not output.exists()


def test_result_and_log_tampering_are_detected(
    locked_protocol: LockedProtocol,
    tmp_path: Path,
) -> None:
    config = replace(locked_protocol.config, smoke=SmokeBudget(1, 1, 1, 1))
    protocol = LockedProtocol(locked_protocol.index, config, locked_protocol.path)
    path = _run(protocol, tmp_path / "runs")
    result = read_sealed(path / "result.json")
    result["encoder_unchanged"] = False
    (path / "result.json").unlink()
    write_json(path / "result.json", seal(result))
    with pytest.raises(ValueError, match="artifact changed"):
        read_probe_run(path)


def test_released_test_blocks_further_probe_calibration(
    locked_protocol: LockedProtocol,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Avoiding mutating the shared fixture: simulate the release marker at its exact lookup.
    original = Path.exists

    def exists(path: Path) -> bool:
        return path.name == "test_release.json" or original(path)

    monkeypatch.setattr(Path, "exists", exists)
    with pytest.raises(ValueError, match="already released"):
        _run(locked_protocol, tmp_path / "blocked")


def test_missing_official_weights_never_fall_back_to_random(
    locked_protocol: LockedProtocol,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="requires a local official checkpoint"):
        _run(locked_protocol, tmp_path / "blocked", arm="suprem_frozen")


def test_zero_test_array_access_is_enforced_by_actual_loader(
    locked_protocol: LockedProtocol,
    baseline_config: BaselineConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(baseline_config, smoke=SmokeBudget(1, 1, 1, 1))
    protocol = LockedProtocol(locked_protocol.index, config, locked_protocol.path)
    original = protocol.index.load
    test_ids = set(protocol.partition_ids("test"))

    def guarded_load(identifier: str):
        if identifier in test_ids:
            raise AssertionError("A held-out image/mask/QC was requested.")
        return original(identifier)

    monkeypatch.setattr(protocol.index, "load", guarded_load)
    path = _run(protocol, tmp_path / "guarded")
    result, _ = read_probe_run(path)
    assert result["test_access"]["read_counts"]["test"] == 0


def test_real_data_policy_never_reads_corrupted_test_images_masks_or_qc(
    locked_protocol: LockedProtocol,
    tmp_path: Path,
) -> None:
    root = tmp_path / "private"
    shutil.copytree(locked_protocol.index.root, root)
    payload = read_sealed(root / "index.json")
    payload["synthetic"] = False  # Synthetic arrays exercise the real-data access POLICY.
    (root / "index.json").unlink()
    write_json(root / "index.json", seal(payload))
    index = CandidateIndex(root / "index.json")
    config = replace(locked_protocol.config, smoke=SmokeBudget(1, 1, 1, 1))
    partitions = tmp_path / "real-policy-partitions.json"
    create_partition(index, config, partitions)
    protocol = LockedProtocol(index, config, partitions)
    for identifier in protocol.partition_ids("test"):
        record = index.by_id[identifier]
        for name in (record.image_path, record.mask_path, record.target_qc_path):
            (root / name).write_bytes(b"corrupt held-out bytes: access would fail")
    result, _ = read_probe_run(_run(protocol, tmp_path / "real-policy"))
    assert result["synthetic"] is False and result["test"] is None
    assert result["test_access"]["read_counts"]["test"] == 0


def test_nonfinite_loss_preserves_failure_evidence_and_never_completes(
    locked_protocol: LockedProtocol,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = training.DiceCELoss

    class BadLoss(original):
        def forward(self, *args: object, **kwargs: object):
            return super().forward(*args, **kwargs) * float("nan")

    monkeypatch.setattr(training, "DiceCELoss", BadLoss)
    with pytest.raises(RuntimeError, match="Non-finite probe loss"):
        _run(locked_protocol, tmp_path / "failed")
    path = tmp_path / "failed/random_frozen/smoke/k01/head991"
    assert (path / "FAILED.json").exists() and not (path / "COMPLETED.json").exists()


def test_source_drift_during_run_cannot_produce_completed_evidence(
    locked_protocol: LockedProtocol,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(locked_protocol.config, smoke=SmokeBudget(1, 1, 1, 1))
    protocol = LockedProtocol(locked_protocol.index, config, locked_protocol.path)
    original = training.evaluate

    def drift_after_final_validation(*args: object, **kwargs: object):
        result = original(*args, **kwargs)
        if args[1].state == "locked":
            monkeypatch.setattr(training, "implementation_fingerprint", lambda: "changed")
        return result

    monkeypatch.setattr(training, "evaluate", drift_after_final_validation)
    with pytest.raises(RuntimeError, match="Implementation changed"):
        _run(protocol, tmp_path / "drift")
    assert not (tmp_path / "drift/random_frozen/smoke/k01/head991/COMPLETED.json").exists()


@pytest.mark.parametrize(
    "statuses",
    [
        ("passed", "passed"),
        ("passed", "blocked"),
        ("blocked", "passed"),
        ("blocked", "blocked"),
    ],
)
def test_underoptimized_pairs_do_not_emit_a_performance_difference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    statuses: tuple[str, str],
) -> None:
    common = dict.fromkeys(
        (
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
        ),
        "identical synthetic contract fixture",
    )
    common_model = dict.fromkeys(
        (
            "head_initial_sha256",
            "encoder_specification",
            "parameter_budget",
        ),
        "identical synthetic model fixture",
    )
    a, b = tmp_path / "pretrained", tmp_path / "random"
    for path, status in zip((a, b), statuses, strict=True):
        write_json(path / "optimization_gate.json", seal({"status": status}))
        write_json(path / "result.json", {"fixture": "not a real experiment"})

    def fixture_read(path: Path):
        pretrained = path == a
        return (
            {"validation": {"patient_macro": {"dice": 0.6 if pretrained else 0.4}}},
            {
                **common,
                "arm": "suprem_frozen" if pretrained else "random_frozen",
                "model": {
                    **common_model,
                    "checkpoint_loading": {"fixture": True} if pretrained else None,
                },
            },
        )

    monkeypatch.setattr(training, "read_probe_run", fixture_read)
    result = compare_probes(a, b, tmp_path / "comparison.json")
    adequate = statuses == ("passed", "passed")
    assert result["performance_comparison_interpretable"] is adequate
    if adequate:
        assert result["validation_patient_macro_dice_difference"] == pytest.approx(0.2)
    else:
        assert result["validation_patient_macro_dice_difference"] is None
