"""Random-only SegResNet Construction and Reversible Stride Padding."""

from __future__ import annotations

import hashlib
from contextlib import ExitStack
from dataclasses import asdict
from typing import Any
from unittest.mock import patch

import torch
import torch.nn.functional as F
from monai.networks.nets.segresnet import SegResNet
from torch import nn

from medfm_adapt3d.adaptation.parameter_audit import audit_parameter_budget
from medfm_adapt3d.baseline.config import ModelConfig
from medfm_adapt3d.engineering.reproducibility import configure_reproducibility


def state_fingerprint(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class AlignedSegResNet(nn.Module):
    """Keeping Commit-03 odd-sized crops intact: pad to stride 8, then remove padding."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.network = SegResNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            init_filters=config.init_filters,
            blocks_down=config.blocks_down,
            blocks_up=config.blocks_up,
            norm=("GROUP", {"num_groups": config.norm_groups}),
            upsample_mode=config.upsample_mode,
            dropout_prob=None,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 5 or image.shape[1] != 1:
            raise ValueError("SegResNet input must be [B,1,D,H,W].")
        shape = image.shape[-3:]
        padding = tuple(v for size in reversed(shape) for v in (0, (-size) % 8))
        logits = self.network(F.pad(image, padding, mode="replicate"))
        logits = logits[..., : shape[0], : shape[1], : shape[2]]
        if logits.shape != image.shape:
            raise RuntimeError("SegResNet logits no longer align with the input physical grid.")
        return logits


def build_random_model(
    config: ModelConfig, model_seed: int
) -> tuple[AlignedSegResNet, dict[str, Any]]:
    """Block checkpoint/URL weight loads during construction, then fingerprint all weights.

    Construction is single-process. The temporary guards cover the supported
    PyTorch loading routes; they are not a sandbox against arbitrary modified code.
    Restoring the run's own selected checkpoint occurs later, after its lock.
    """
    if config.initialization != "random" or config.architecture != "SegResNet":
        raise ValueError("Only the conventional random SegResNet baseline can be constructed.")
    configure_reproducibility(model_seed, use_deterministic_algorithms=True)
    with ExitStack() as guards:
        forbidden = RuntimeError(
            "Pretrained/checkpoint loading is forbidden during initialization."
        )
        for owner, name in (
            (torch, "load"),
            (nn.Module, "load_state_dict"),
            (torch.hub, "load"),
            (torch.hub, "load_state_dict_from_url"),
        ):
            guards.enter_context(patch.object(owner, name, side_effect=forbidden))
        model = AlignedSegResNet(config)
    budget = audit_parameter_budget(model)
    if budget.total_parameters <= 0 or budget.trainable_fraction != 1.0:
        raise RuntimeError("Every intended conventional-model parameter must be trainable.")
    audit = {
        **asdict(budget),
        "fraction_trainable": budget.trainable_fraction,
        "initialization": "random",
        "weights_loaded_during_initialization": False,
        "initial_state_sha256": state_fingerprint(model),
        "model_seed": model_seed,
        "architecture": asdict(config),
        "padding": "right-replicate-to-8-and-unpad",
    }
    return model, audit
