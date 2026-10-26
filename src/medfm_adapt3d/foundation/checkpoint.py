"""Strict Local Loading: Every Encoder Tensor Must Be Present, Compatible, and Finite."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from medfm_adapt3d.baseline.artifacts import file_hash

ENCODER_PREFIXES = ("convInit.", "down_layers.")
DISCARDED_PREFIXES = ("up_layers.", "up_samples.", "conv_final.")


def load_encoder_checkpoint(
    encoder: nn.Module,
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
) -> dict[str, Any]:
    """No network access, permissive unpickling, partial load, or random fallback."""
    if path.stat().st_size != expected_bytes or file_hash(path) != expected_sha256:
        raise ValueError("Checkpoint size/SHA-256 mismatch; nothing was deserialized.")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("net"), dict):
        raise ValueError("Expected the released SuPreM dictionary containing 'net'.")
    source = payload["net"]
    if not source or not all(isinstance(key, str) for key in source):
        raise ValueError("Checkpoint tensor names must be non-empty strings.")
    prefixed = [key.startswith("module.") for key in source]
    if any(prefixed) and not all(prefixed):
        raise ValueError("Mixed parallel/non-parallel checkpoint prefixes are ambiguous.")
    selected, discarded = {}, []
    for raw_key, value in source.items():
        key = raw_key.removeprefix("module.")
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError(f"Non-tensor or non-finite checkpoint value: {key}.")
        if key.startswith(ENCODER_PREFIXES):
            selected[key] = value
        elif key.startswith(DISCARDED_PREFIXES):
            discarded.append(key)
        else:
            raise ValueError(f"Unrecognized checkpoint namespace: {key}.")
    expected = encoder.state_dict()
    if set(selected) != set(expected):
        raise ValueError("Encoder key coverage must be exactly 100%; missing/extra keys rejected.")
    for key, value in selected.items():
        if value.shape != expected[key].shape or value.dtype != expected[key].dtype:
            raise ValueError(f"Encoder shape/dtype mismatch: {key}.")
    if file_hash(path) != expected_sha256:
        raise ValueError("Checkpoint bytes changed while loading.")
    encoder.load_state_dict(selected, strict=True)
    return {
        "checkpoint_sha256": expected_sha256,
        "checkpoint_size_bytes": expected_bytes,
        "encoder_tensor_count": len(selected),
        "encoder_tensor_coverage": 1.0,
        "encoder_element_count": sum(value.numel() for value in selected.values()),
        "discarded_tensor_names": sorted(discarded),
        "deserialization": "torch.load(weights_only=True,map_location=cpu)",
        "strict": True,
    }
