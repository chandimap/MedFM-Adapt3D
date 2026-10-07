from dataclasses import replace

import pytest
import torch
from torch import nn

from medfm_adapt3d.baseline.config import BaselineConfig
from medfm_adapt3d.baseline.model import build_random_model, state_fingerprint
from medfm_adapt3d.baseline.protocol import LockedProtocol
from medfm_adapt3d.baseline.sampling import PatientBalancedSampler, augment_pair


def test_five_independent_reproducible_initializations(baseline_config: BaselineConfig) -> None:
    fingerprints = []
    for seed in baseline_config.seeds.model:
        model, audit = build_random_model(baseline_config.model, seed)
        assert audit["initialization"] == "random"
        assert audit["weights_loaded_during_initialization"] is False
        assert audit["total_parameters"] == audit["trainable_parameters"] > 0
        assert audit["fraction_trainable"] == 1.0
        assert all(p.requires_grad for p in model.parameters())
        fingerprints.append(state_fingerprint(model))
    assert len(set(fingerprints)) == 5
    repeated, _ = build_random_model(baseline_config.model, baseline_config.seeds.model[0])
    assert state_fingerprint(repeated) == fingerprints[0]


@pytest.mark.parametrize("loading_route", ["torch.load", "load_state_dict", "hub"])
def test_unexpected_weight_loading_is_blocked(
    baseline_config: BaselineConfig,
    monkeypatch: pytest.MonkeyPatch,
    loading_route: str,
) -> None:
    def corrupted_constructor(**kwargs: object) -> nn.Module:
        if loading_route == "torch.load":
            torch.load("forbidden.pt")
        elif loading_route == "hub":
            torch.hub.load_state_dict_from_url("https://invalid.example/forbidden.pt")
        else:
            nn.Linear(1, 1).load_state_dict({})
        raise AssertionError("Loading guard failed.")

    monkeypatch.setattr("medfm_adapt3d.baseline.model.SegResNet", corrupted_constructor)
    with pytest.raises(RuntimeError, match="loading is forbidden"):
        build_random_model(baseline_config.model, 17)


def test_unexpected_frozen_parameters_rejected(
    baseline_config: BaselineConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def frozen_constructor(**kwargs: object) -> nn.Module:
        model = nn.Conv3d(1, 1, 1)
        model.requires_grad_(False)
        return model

    monkeypatch.setattr("medfm_adapt3d.baseline.model.SegResNet", frozen_constructor)
    with pytest.raises(RuntimeError, match="must be trainable"):
        build_random_model(baseline_config.model, 17)


@pytest.mark.parametrize("size", [17, 65])
def test_network_preserves_odd_crop_shape_and_gradient(
    baseline_config: BaselineConfig, size: int
) -> None:
    model, _ = build_random_model(baseline_config.model, 17)
    image = torch.zeros(1, 1, size, size, size, requires_grad=True)
    output = model(image)
    assert output.shape == image.shape
    output.mean().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()
    assert all(p.grad is not None for p in model.parameters())


def test_patient_balance_uses_all_reader_targets(locked_protocol: LockedProtocol) -> None:
    records = [locked_protocol.index.by_id[i] for i in locked_protocol.support_ids(5)]
    first = PatientBalancedSampler(records, 401)
    second = PatientBalancedSampler(records, 401)
    ids = [first.next_id() for _ in range(50)]
    assert ids == [second.next_id() for _ in range(50)]
    assert set(first.patient_exposure.values()) == {10}
    assert all(n > 0 for n in first.target_exposure.values())
    for offset in range(0, len(ids), 5):
        assert (
            len({locked_protocol.index.by_id[i].patient_id for i in ids[offset : offset + 5]}) == 5
        )


def test_identity_augmentation_preserves_anatomy_labels_and_independent_rng() -> None:
    mask = torch.zeros(1, 9, 9, 9, dtype=torch.uint8)
    mask[0, 2, 3, 4] = 1
    image = mask.float()
    generator = torch.Generator().manual_seed(307)
    original_rng = generator.get_state().clone()
    first = augment_pair(image, mask, generator=generator, probability=0)
    torch.manual_seed(999)
    second = augment_pair(image, mask, generator=torch.Generator().manual_seed(71), probability=0)
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
    assert torch.equal(first[0], first[1].float())
    assert first[1].sum() == mask.sum()
    assert torch.equal(first[0], image) and torch.equal(first[1], mask)
    assert torch.equal(generator.get_state(), original_rng)


@pytest.mark.parametrize("probability", [0.1, 0.5, 1.0])
def test_spatial_reflections_are_forbidden(
    baseline_config: BaselineConfig, probability: float
) -> None:
    with pytest.raises(ValueError, match="disables all spatial"):
        replace(
            baseline_config,
            sampling=replace(baseline_config.sampling, flip_probability=probability),
        )
    with pytest.raises(ValueError, match="no spatial reflections"):
        augment_pair(
            torch.zeros(1, 3, 3, 3),
            torch.zeros(1, 3, 3, 3),
            generator=torch.Generator().manual_seed(307),
            probability=probability,
        )


def test_parameter_policy_cannot_be_changed_to_pretrained(baseline_config: BaselineConfig) -> None:
    with pytest.raises(ValueError, match="random"):
        build_random_model(replace(baseline_config.model, initialization="pretrained"), 17)
