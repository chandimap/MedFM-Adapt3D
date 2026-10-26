from pathlib import Path

import pytest
import torch
from monai.networks.nets import SegResNet

from medfm_adapt3d.baseline.model import state_fingerprint
from medfm_adapt3d.foundation.model import (
    DenseLinearProbe,
    FrozenRepresentation,
    SegResNetEncoder,
    build_frozen_model,
)
from medfm_adapt3d.foundation.provenance import load_model_record


def _random(seed: int = 991):
    return build_frozen_model(
        load_model_record(Path("configs/foundation/suprem_segresnet.yaml")),
        arm="random_frozen",
        head_seed=seed,
    )


def test_encoder_matches_monai_feature_semantics() -> None:
    reference = SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        init_filters=16,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        norm=("GROUP", {"num_groups": 8}),
        dropout_prob=None,
    )
    encoder = SegResNetEncoder()
    encoder.load_state_dict(
        {n: reference.state_dict()[n] for n in encoder.state_dict()}, strict=True
    )
    image = torch.randn(1, 1, 24, 24, 24)
    with torch.no_grad():
        _, expected = reference.encode(image)
        actual = encoder(image)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_head_updates_without_encoder_gradients_or_state_changes() -> None:
    model, audit = _random()
    initial_head = state_fingerprint(model.head)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=0.01)
    model.train()
    assert model.head.training and not model.encoder.training
    image = torch.randn(1, 1, 17, 19, 21)
    model(image).square().mean().backward()
    assert all(p.grad is None for p in model.encoder.parameters())
    assert all(p.grad is not None for p in model.head.parameters())
    optimizer.step()
    model.assert_frozen(audit["encoder_initial_sha256"])
    assert state_fingerprint(model.head) != initial_head
    assert audit["parameter_budget"]["trainable_parameters"] == 241


def test_reproducibility_and_separate_representation_head_rngs() -> None:
    a, aa = _random(991)
    b, bb = _random(991)
    c, cc = _random(997)
    assert state_fingerprint(a) == state_fingerprint(b)
    assert aa["encoder_initial_sha256"] == cc["encoder_initial_sha256"]
    assert aa["head_initial_sha256"] == bb["head_initial_sha256"]
    assert aa["head_initial_sha256"] != cc["head_initial_sha256"]
    assert state_fingerprint(a.encoder) == state_fingerprint(c.encoder)


@pytest.mark.parametrize("shape", [(16, 16, 16), (17, 19, 21), (25, 17, 24)])
def test_dense_probe_preserves_full_input_grid(shape: tuple[int, ...]) -> None:
    model, _ = _random()
    with torch.no_grad():
        for layer in model.head.projections:
            layer.weight.zero_()
        model.head.bias.fill_(2)
    result = model(torch.zeros(1, 1, *shape))
    assert result.shape == (1, 1, *shape)
    torch.testing.assert_close(result, torch.full_like(result, 2), rtol=0, atol=0)


@pytest.mark.parametrize("fault", ["requires_grad", "weight", "mode"])
def test_freeze_violation_is_detected(fault: str) -> None:
    model, audit = _random()
    if fault == "requires_grad":
        next(model.encoder.parameters()).requires_grad_(True)
    elif fault == "mode":
        model.encoder.train()
    else:
        with torch.no_grad():
            next(model.encoder.parameters()).add_(1)
    with pytest.raises(RuntimeError):
        model.assert_frozen(audit["encoder_initial_sha256"])


@pytest.mark.parametrize("shape", [(1, 2, 16, 16, 16), (1, 1, 7, 16, 16), (1, 16, 16, 16)])
def test_invalid_input_shape_rejected(shape: tuple[int, ...]) -> None:
    model = FrozenRepresentation(SegResNetEncoder(), DenseLinearProbe())
    with pytest.raises(ValueError, match="Expected"):
        model(torch.zeros(shape))


def test_nonfinite_input_rejected() -> None:
    model, _ = _random()
    with pytest.raises(ValueError, match="finite"):
        model(torch.full((1, 1, 16, 16, 16), float("nan")))


def test_random_control_does_not_deserialize_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("A matched random control must not load a checkpoint.")

    monkeypatch.setattr(torch, "load", forbidden)
    _, audit = _random()
    assert audit["checkpoint_loading"] is None
