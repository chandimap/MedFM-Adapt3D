from pathlib import Path

import pytest
import torch

from medfm_adapt3d.baseline.artifacts import file_hash
from medfm_adapt3d.baseline.model import state_fingerprint
from medfm_adapt3d.foundation.checkpoint import load_encoder_checkpoint
from medfm_adapt3d.foundation.model import SegResNetEncoder


def _save(path: Path, payload: object) -> dict[str, object]:
    torch.save(payload, path)
    return {"expected_sha256": file_hash(path), "expected_bytes": path.stat().st_size}


def test_complete_parallel_encoder_load_is_exact(tmp_path: Path) -> None:
    source, target = SegResNetEncoder(), SegResNetEncoder()
    state = {f"module.{n}": v for n, v in source.state_dict().items()}
    state["module.conv_final.2.conv.weight"] = torch.ones(32, 16, 1, 1, 1)
    path = tmp_path / "synthetic-checkpoint.pth"
    expected = _save(path, {"net": state})
    audit = load_encoder_checkpoint(target, path, **expected)
    assert state_fingerprint(target) == state_fingerprint(source)
    assert audit["encoder_tensor_coverage"] == 1
    assert audit["encoder_tensor_count"] == 58
    assert audit["encoder_element_count"] == 4399344
    assert audit["discarded_tensor_names"] == ["conv_final.2.conv.weight"]


@pytest.mark.parametrize("fault", ["missing", "unknown", "shape", "dtype", "nan", "mixed"])
def test_invalid_checkpoint_cannot_partially_initialize_encoder(tmp_path: Path, fault: str) -> None:
    encoder = SegResNetEncoder()
    initial = state_fingerprint(encoder)
    state = {n: v.clone() for n, v in encoder.state_dict().items()}
    first = next(iter(state))
    if fault == "missing":
        del state[first]
    elif fault == "unknown":
        state["unrecognized.weight"] = torch.ones(1)
    elif fault == "shape":
        state[first] = state[first][:1]
    elif fault == "dtype":
        state[first] = state[first].double()
    elif fault == "nan":
        state[first].fill_(float("nan"))
    else:
        state[f"module.{first}"] = state.pop(first)
    path = tmp_path / "synthetic-invalid.pth"
    expected = _save(path, {"net": state})
    with pytest.raises(ValueError):
        load_encoder_checkpoint(encoder, path, **expected)
    assert state_fingerprint(encoder) == initial


@pytest.mark.parametrize("fault", ["hash", "size"])
def test_identity_failure_precedes_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    path = tmp_path / "not-a-checkpoint.pth"
    path.write_bytes(b"deliberately invalid checkpoint bytes")
    expected = {"expected_sha256": file_hash(path), "expected_bytes": path.stat().st_size}
    expected["expected_sha256" if fault == "hash" else "expected_bytes"] = (
        "0" * 64 if fault == "hash" else 0
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Deserialization must never start after identity failure.")

    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(ValueError, match="nothing was deserialized"):
        load_encoder_checkpoint(SegResNetEncoder(), path, **expected)


@pytest.mark.parametrize("payload", [{}, {"net": {}}, {"net": {1: torch.ones(1)}}, {"net": []}])
def test_wrong_checkpoint_container_rejected(tmp_path: Path, payload: object) -> None:
    path = tmp_path / "synthetic-container.pth"
    expected = _save(path, payload)
    with pytest.raises(ValueError):
        load_encoder_checkpoint(SegResNetEncoder(), path, **expected)
