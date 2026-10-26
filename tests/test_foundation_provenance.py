from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from medfm_adapt3d.foundation.provenance import claim_gate, load_model_record


def _record():
    return load_model_record(Path("configs/foundation/suprem_segresnet.yaml"))


@pytest.mark.parametrize("scope", ["clean_generalization", "clinical_readiness"])
def test_unresolved_overlap_blocks_strong_claims(scope: str) -> None:
    with pytest.raises(ValueError, match="overlap is unresolved"):
        claim_gate(_record(), scope)


def test_engineering_is_allowed_but_real_test_is_closed() -> None:
    gate = claim_gate(_record())
    assert gate["allowed"] and not gate["real_held_out_test_enabled"]
    assert gate["overlap_status"] == "NOT_ESTABLISHED"


@pytest.mark.parametrize(
    "field,value",
    [
        ("overlap_status", "DISJOINT"),
        ("checkpoint_sha256", "0" * 64),
        ("checkpoint_revision", "main"),
        ("license_status", "UNRESTRICTED"),
    ],
)
def test_claim_and_checkpoint_identity_cannot_be_changed_by_a_config_flag(
    field: str, value: str
) -> None:
    with pytest.raises(ValueError):
        replace(_record(), **{field: value})


def test_unknown_configuration_fields_rejected(tmp_path: Path) -> None:
    payload = yaml.safe_load(Path("configs/foundation/suprem_segresnet.yaml").read_text())
    payload["allow_overlap"] = True
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="schema exactly"):
        load_model_record(path)
