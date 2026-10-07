from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from medfm_adapt3d.baseline.artifacts import read_json, read_sealed, seal, write_json
from medfm_adapt3d.baseline.candidates import CandidateIndex
from medfm_adapt3d.baseline.config import BaselineConfig, load_baseline_config
from medfm_adapt3d.baseline.protocol import LockedProtocol, create_partition, expected_partition
from medfm_adapt3d.data.splits import ProtocolViolation


def test_patient_disjoint_nested_patient_budgets(locked_protocol: LockedProtocol) -> None:
    groups = locked_protocol.payload["patients"]
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert set(groups[left]).isdisjoint(groups[right])
    previous: set[str] = set()
    for k in (1, 5, 10, 20):
        identifiers = locked_protocol.support_ids(k)
        patients = {locked_protocol.index.by_id[i].patient_id for i in identifiers}
        assert len(patients) == k
        assert previous < patients
        assert set(identifiers) == {
            r.candidate_id for r in locked_protocol.index.records if r.patient_id in patients
        }
        previous = patients
    assert len(locked_protocol.support_ids(20)) > 20  # Readers do not increase K.


def test_model_seeds_cannot_change_partition_or_support(
    candidate_index: CandidateIndex,
    baseline_config: BaselineConfig,
) -> None:
    changed = replace(baseline_config, seeds=replace(baseline_config.seeds, model=(2, 3, 5, 7, 11)))
    assert expected_partition(candidate_index, baseline_config) == expected_partition(
        candidate_index, changed
    )


def test_pilot_k_is_never_silently_reduced(
    candidate_index: CandidateIndex,
    baseline_config: BaselineConfig,
    tmp_path: Path,
) -> None:
    payload = read_sealed(candidate_index.path)
    patients = set(candidate_index.patients[:10])
    payload["records"] = [r for r in payload["records"] if r["patient_id"] in patients]
    path = candidate_index.root / "unused"  # Arrays are not read for partition construction.
    subset = tmp_path / "index.json"
    write_json(subset, seal(payload))
    index = CandidateIndex(subset)
    partition = tmp_path / "split.json"
    create_partition(index, baseline_config, partition)
    protocol = LockedProtocol(index, baseline_config, partition)
    assert len(protocol.payload["patients"]["train"]) == 6
    assert len({index.by_id[i].patient_id for i in protocol.support_ids(1)}) == 1
    assert len({index.by_id[i].patient_id for i in protocol.support_ids(5)}) == 5
    for k in (10, 20):
        with pytest.raises(ProtocolViolation, match=f"Requested K={k}.*only 6"):
            protocol.support_ids(k)
    assert not path.exists()


def test_resealed_tampered_partition_still_rejected(
    locked_protocol: LockedProtocol,
    tmp_path: Path,
) -> None:
    payload = {
        **locked_protocol.payload,
        "patients": {
            key: list(value) for key, value in locked_protocol.payload["patients"].items()
        },
    }
    payload["patients"]["test"][0] = payload["patients"]["train"][0]
    path = tmp_path / "tampered.json"
    write_json(path, seal(payload))
    with pytest.raises(ProtocolViolation, match="Tampered partition"):
        LockedProtocol(locked_protocol.index, locked_protocol.config, path)


def test_candidate_index_checksum_rejects_edits(
    candidate_index: CandidateIndex, tmp_path: Path
) -> None:
    payload = read_json(candidate_index.path)
    payload["records"][0]["patient_id"] = "injected-patient"
    path = tmp_path / "corrupt.json"
    write_json(path, payload)
    with pytest.raises(ValueError, match="Integrity failure"):
        CandidateIndex(path)


def test_resealed_lineage_tampering_rejected(
    candidate_index: CandidateIndex, tmp_path: Path
) -> None:
    payload = read_sealed(candidate_index.path)
    payload["records"][0]["patient_id"] = "injected-patient"
    path = tmp_path / "corrupt.json"
    write_json(path, seal(payload))
    with pytest.raises(ValueError, match="lineage/identity"):
        CandidateIndex(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("checkpoint", "external.pt"),
        ("pretrained", True),
        ("init_filters", 15),
    ],
)
def test_config_rejects_weights_and_invalid_model_fields(
    field: str, value: object, tmp_path: Path
) -> None:
    payload = yaml.safe_load(Path("configs/baseline/random_segresnet.yaml").read_text())
    payload["model"][field] = value
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError):
        load_baseline_config(path)


def test_recipe_requires_five_separate_definitive_seeds(baseline_config: BaselineConfig) -> None:
    with pytest.raises(ValueError, match="five distinct"):
        replace(baseline_config, seeds=replace(baseline_config.seeds, model=(17, 29)))
    with pytest.raises(ValueError, match="Engineering seeds"):
        replace(baseline_config, seeds=replace(baseline_config.seeds, calibration=17))


@pytest.mark.parametrize(
    "patients,expected", [(3, (1,)), (10, (1, 5)), (17, (1, 10)), (34, (1, 20))]
)
def test_calibration_uses_smallest_and_largest_feasible_budget(
    candidate_index: CandidateIndex,
    baseline_config: BaselineConfig,
    tmp_path: Path,
    patients: int,
    expected: tuple[int, ...],
) -> None:
    payload = read_sealed(candidate_index.path)
    selected = set(candidate_index.patients[:patients])
    payload["records"] = [r for r in payload["records"] if r["patient_id"] in selected]
    path = tmp_path / "index.json"
    write_json(path, seal(payload))
    index = CandidateIndex(path)
    partition = tmp_path / "partitions.json"
    create_partition(index, baseline_config, partition)
    protocol = LockedProtocol(index, baseline_config, partition)
    assert protocol.calibration_budgets == expected
