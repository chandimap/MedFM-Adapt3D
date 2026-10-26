"""One Volumetric Encoder Family and a 241-parameter Multiscale Dense Linear Probe."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from monai.networks.nets import SegResNet
from torch import nn

from medfm_adapt3d.adaptation.parameter_audit import audit_parameter_budget
from medfm_adapt3d.baseline.model import state_fingerprint
from medfm_adapt3d.engineering.reproducibility import configure_reproducibility
from medfm_adapt3d.foundation.checkpoint import load_encoder_checkpoint
from medfm_adapt3d.foundation.provenance import ModelRecord, claim_gate

ARMS = ("suprem_frozen", "random_frozen")
REPRESENTATION_SEED = 887
WIDTHS = (16, 32, 64, 128)
ENCODER_SPEC = {
    "architecture": "MONAI SegResNet encoder",
    "init_filters": 16,
    "blocks_down": [1, 2, 2, 4],
    "normalization": "GroupNorm(8)",
    "dropout": None,
    "in_channels": 1,
}


class SegResNetEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        network = SegResNet(
            spatial_dims=3,
            init_filters=16,
            in_channels=1,
            out_channels=1,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            norm=("GROUP", {"num_groups": 8}),
            dropout_prob=None,
        )
        self.convInit = network.convInit
        self.down_layers = network.down_layers

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        value = self.convInit(image)
        features = []
        for stage in self.down_layers:
            value = stage(value)
            features.append(value)
        return tuple(features)


class DenseLinearProbe(nn.Module):
    """Projecting each scale to one logit, then interpolating logits rather than 240 channels."""

    def __init__(self) -> None:
        super().__init__()
        self.projections = nn.ModuleList(nn.Conv3d(w, 1, 1, bias=False) for w in WIDTHS)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, features: tuple[torch.Tensor, ...], shape: tuple[int, ...]) -> torch.Tensor:
        if len(features) != len(self.projections):
            raise ValueError("All four prespecified encoder feature scales are required.")
        logits = [
            F.interpolate(layer(feature), size=shape, mode="trilinear", align_corners=False)
            for layer, feature in zip(self.projections, features, strict=True)
        ]
        return torch.stack(logits).sum(dim=0) + self.bias.view(1, 1, 1, 1, 1)


class FrozenRepresentation(nn.Module):
    def __init__(self, encoder: SegResNetEncoder, head: DenseLinearProbe) -> None:
        super().__init__()
        self.encoder = encoder.requires_grad_(False)
        self.head = head
        self.encoder.eval()

    def train(self, mode: bool = True) -> FrozenRepresentation:
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 5 or image.shape[1] != 1 or min(image.shape[-3:]) < 8:
            raise ValueError("Expected [B,1,D,H,W] with spatial dimensions >= 8.")
        if not torch.isfinite(image).all():
            raise ValueError("Encoder input must be finite.")
        shape = image.shape[-3:]
        padding = tuple(v for size in reversed(shape) for v in (0, (-size) % 8))
        padded = F.pad(image, padding, mode="replicate")
        # no_grad keeps ordinary tensors that the trainable head can save for backward.
        with torch.no_grad():
            features = self.encoder(padded)
        logits = self.head(features, tuple(padded.shape[-3:]))
        return logits[..., : shape[0], : shape[1], : shape[2]]

    def assert_frozen(self, expected_sha256: str) -> None:
        if self.encoder.training or any(
            p.requires_grad or p.grad is not None for p in self.encoder.parameters()
        ):
            raise RuntimeError("Encoder freeze contract was violated.")
        if state_fingerprint(self.encoder) != expected_sha256:
            raise RuntimeError("Encoder parameters/buffers changed during head adaptation.")


def build_frozen_model(
    record: ModelRecord,
    *,
    arm: str,
    head_seed: int,
    checkpoint: Path | None = None,
) -> tuple[FrozenRepresentation, dict[str, Any]]:
    claim_gate(record)
    if arm not in ARMS:
        raise ValueError("Only the prespecified pretrained and matched random frozen arms exist.")
    if type(head_seed) is not int or not 0 <= head_seed < 2**32:
        raise ValueError("Head seed must be a non-negative 32-bit integer.")
    if (arm == "suprem_frozen") != (checkpoint is not None):
        raise ValueError("Only the pretrained arm requires a local official checkpoint.")
    configure_reproducibility(REPRESENTATION_SEED, use_deterministic_algorithms=True)
    encoder = SegResNetEncoder()
    loading = None
    if checkpoint is not None:
        loading = load_encoder_checkpoint(
            encoder,
            checkpoint,
            expected_sha256=record.checkpoint_sha256,
            expected_bytes=record.checkpoint_size_bytes,
        )
    configure_reproducibility(head_seed, use_deterministic_algorithms=True)
    head = DenseLinearProbe()
    model = FrozenRepresentation(encoder, head)
    budget = audit_parameter_budget(model)
    if budget.trainable_parameters != 241:
        raise RuntimeError("Only the prespecified 241 head parameters may be trainable.")
    encoder_hash = state_fingerprint(encoder)
    model.assert_frozen(encoder_hash)
    return model, {
        "arm": arm,
        "encoder_specification": ENCODER_SPEC,
        "representation_seed": REPRESENTATION_SEED if loading is None else None,
        "head_seed": head_seed,
        "head_initial_sha256": state_fingerprint(head),
        "encoder_initial_sha256": encoder_hash,
        "parameter_budget": asdict(budget),
        "checkpoint_loading": loading,
        "head": "four-scale 1x1x1 linear logit projections plus one bias",
        "padding": "right-replicate-to-8-and-unpad",
        "feature_resize": "trilinear-logits; align_corners=False",
        "input_policy": (
            "unchanged indexed thoracic preprocessing; no hidden model-specific transform"
        ),
        "source_input_recipe_compatibility": "NOT_ESTABLISHED",
    }
