"""One nonlinear residual adaptation of the finest frozen SegResNet output."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from medfm_adapt3d.adaptation.parameter_audit import audit_parameter_budget
from medfm_adapt3d.baseline.model import state_fingerprint
from medfm_adapt3d.foundation.model import WIDTHS, DenseLinearProbe, FrozenRepresentation

# Factorial controls isolate the method within each representation source.
FEATURE_ARMS = ("suprem_frozen", "suprem_feature", "random_frozen", "random_feature")


@dataclass(frozen=True)
class FeatureRecipe:
    method: str
    feature_level: int
    bottleneck: int
    adapter_seed: int

    def __post_init__(self) -> None:
        if self.method != "residual_pointwise_relu":
            raise ValueError("Unsupported feature adaptation method.")
        if type(self.feature_level) is not int or self.feature_level != 0:
            raise ValueError("Only the full-resolution encoder output at level zero is supported.")
        if type(self.bottleneck) is not int or not 0 < self.bottleneck < WIDTHS[0]:
            raise ValueError("Bottleneck width must be a positive integer below 16.")
        if type(self.adapter_seed) is not int or not 0 <= self.adapter_seed < 2**32:
            raise ValueError("Adapter seed must be an unsigned 32-bit integer.")

    @classmethod
    def load(cls, path: Path) -> FeatureRecipe:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or set(data) != {f.name for f in fields(cls)}:
            raise ValueError("Feature recipe has missing or unknown fields.")
        return cls(**data)


class ResidualFeatureProbe(nn.Module):
    """Adapt stage-zero output outside the frozen graph, then use the original probe."""

    def __init__(self, probe: DenseLinearProbe, recipe: FeatureRecipe) -> None:
        super().__init__()
        self.probe = probe
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(recipe.adapter_seed)
            self.down = nn.Conv3d(WIDTHS[0], recipe.bottleneck, 1, bias=False)
            self.up = nn.Conv3d(recipe.bottleneck, WIDTHS[0], 1, bias=False)
            nn.init.zeros_(self.up.weight)

    def forward(self, features: tuple[torch.Tensor, ...], shape: tuple[int, ...]) -> torch.Tensor:
        if len(features) != len(WIDTHS):
            raise ValueError("Expected all four SegResNet outputs.")
        fine = features[0]
        if fine.ndim != 5 or fine.shape[1] != WIDTHS[0]:
            raise ValueError("Expected four SegResNet outputs with 16 channels at level zero.")
        adapted = fine + self.up(torch.relu(self.down(fine)))
        return self.probe((adapted, *features[1:]), shape)


def trainable_groups(model: FrozenRepresentation) -> dict[str, nn.Module]:
    head = model.head
    if isinstance(head, ResidualFeatureProbe):
        return {"probe": head.probe, "down": head.down, "up": head.up}
    return {"probe": head}


def configure_feature_arm(
    model: FrozenRepresentation, initialization: dict[str, Any], arm: str, recipe: FeatureRecipe
) -> None:
    """Enforce origin, trainability and measured parameter accounting for all four arms."""
    if arm not in FEATURE_ARMS or arm.split("_")[0] != initialization["arm"].split("_")[0]:
        raise ValueError("Feature arm conflicts with the initialized representation.")
    if arm.endswith("_feature"):
        model.head = ResidualFeatureProbe(model.head, recipe)
    groups = trainable_groups(model)
    permitted = {id(p) for module in groups.values() for p in module.parameters()}
    actual = {id(p) for p in model.parameters() if p.requires_grad}
    if permitted != actual:
        raise RuntimeError("Trainable set differs from the task probe and authorized adapter.")
    budget = audit_parameter_budget(model)
    counts = {name: sum(p.numel() for p in m.parameters()) for name, m in groups.items()}
    added = counts.get("up", 0) + counts.get("down", 0)
    expected = 2 * WIDTHS[0] * recipe.bottleneck if arm.endswith("_feature") else 0
    if (
        counts["probe"] != 241
        or added != expected
        or sum(counts.values()) != budget.trainable_parameters
    ):
        raise RuntimeError("Feature/probe parameter accounting failed.")
    initialization.update(
        arm=arm,
        feature_recipe=asdict(recipe),
        trainable_initial_sha256={name: state_fingerprint(m) for name, m in groups.items()},
        parameter_budget={
            **asdict(budget),
            "encoder_parameters": sum(p.numel() for p in model.encoder.parameters()),
            "frozen_parameters": budget.frozen_parameters,
            "probe_parameters": counts["probe"],
            "adapter_parameters": added,
            "added_parameters": added,
            "trainable_percentage": budget.trainable_percentage,
        },
    )
