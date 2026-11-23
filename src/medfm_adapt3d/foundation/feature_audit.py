"""Measurement of task gradients, fixed-threshold failures and CPU resources."""

from __future__ import annotations

import hashlib
import platform
import threading
import time
from pathlib import Path
from typing import Any, Self

import psutil
import torch
from torch import nn

from medfm_adapt3d.baseline.model import state_fingerprint
from medfm_adapt3d.foundation.feature_adapter import trainable_groups
from medfm_adapt3d.foundation.model import FrozenRepresentation


class CpuMemory:
    """Sample this worker's resident bytes every 10 ms, including both interval endpoints."""

    def __init__(self) -> None:
        self.process = psutil.Process()
        self.stop = threading.Event()
        self.peak = self.samples = 0

    def _sample(self) -> int:
        rss = self.process.memory_info().rss
        self.peak = max(self.peak, rss)
        self.samples += 1
        return rss

    def _poll(self) -> None:
        while not self.stop.wait(0.01):
            self._sample()

    def __enter__(self) -> Self:
        self.start = self._sample()
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.finish()

    def finish(self) -> None:
        if not self.stop.is_set():
            self.stop.set()
            self.thread.join()
            self.end = self._sample()

    def report(self) -> dict[str, Any]:
        self.finish()
        cpu = platform.processor() or platform.machine()
        info = Path("/proc/cpuinfo")
        if info.is_file():
            for line in info.read_text(encoding="utf-8").splitlines():
                if line.startswith("model name"):
                    cpu = line.partition(":")[2].strip()
                    break
        return {
            "method": "psutil single worker process RSS, 10 ms sampling plus endpoints",
            "rss_start_bytes": self.start,
            "rss_end_bytes": self.end,
            "peak_observed_rss_bytes": self.peak,
            "samples": self.samples,
            "cpu_model": cpu,
            "logical_cpus_reported_by_os": psutil.cpu_count(),
            "ram_bytes_reported_by_os": psutil.virtual_memory().total,
            "gpu_memory_bytes": None,
            "psutil_version": psutil.__version__,
        }


def output_fingerprint(model: FrozenRepresentation) -> str:
    """Hash logits on a deterministic, label-free replay grid."""
    with torch.no_grad():
        image = torch.linspace(-1, 1, 16**3).reshape(1, 1, 16, 16, 16)
        output = model(image).detach().contiguous()
        return hashlib.sha256(output.numpy().tobytes()).hexdigest()


def _vector(module: nn.Module, *, gradients: bool = False) -> torch.Tensor:
    values = []
    for p in module.parameters():
        value = p.grad if gradients else p
        if value is None:
            raise RuntimeError("A trainable tensor was disconnected from the loss.")
        values.append(value.detach().reshape(-1))
    return torch.cat(values)


class StepAudit:
    def __init__(self, model: FrozenRepresentation) -> None:
        self.groups = trainable_groups(model)
        self.initial = {n: state_fingerprint(m) for n, m in self.groups.items()}
        self.initial_logits = output_fingerprint(model)
        self.active_gradients: set[str] = set()
        self.updated: set[str] = set()
        self.selected_gradients: set[str] = set()
        self.selected_updates: set[str] = set()
        self.features: list[dict[str, Any]] = []
        self.validations: list[dict[str, Any]] = []
        self.training_seconds = self.validation_seconds = 0.0

    def capture_features(
        self, module: nn.Module, inputs: tuple[torch.Tensor, ...], outputs: tuple[torch.Tensor, ...]
    ) -> None:
        """Aggregate first training-forward activations; no images, labels or identifiers."""
        self.features = [
            {"shape": list(f.shape), "mean": float(f.mean()), "sd": float(f.std(unbiased=False))}
            for f in outputs
        ]

    def gradients(self) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        before = {name: _vector(m) for name, m in self.groups.items()}
        norms = {
            name: float(torch.linalg.vector_norm(_vector(m, gradients=True)))
            for name, m in self.groups.items()
        }
        self.active_gradients.update(name for name, norm in norms.items() if norm > 0)
        return before, norms

    def record_step(
        self,
        before: dict[str, torch.Tensor],
        norms: dict[str, float],
        logits: torch.Tensor,
        target: torch.Tensor,
        threshold: float,
        started: float,
    ) -> dict[str, Any]:
        prob = logits.detach().sigmoid()
        positive = target.bool()
        predicted = prob >= threshold
        updates = {
            name: float(torch.linalg.vector_norm(_vector(m) - before[name]))
            for name, m in self.groups.items()
        }
        self.updated.update(name for name, delta in updates.items() if delta > 0)
        row = {
            "gradient_l2_before_clip": norms,
            "parameter_update_l2": updates,
            "target_foreground_fraction": float(positive.float().mean()),
            "predicted_foreground_fraction": float(predicted.float().mean()),
            "empty_prediction_fraction": float((~predicted.flatten(1).any(1)).float().mean()),
            "mean_probability_on_target": float(prob[positive].mean()) if positive.any() else None,
            "mean_probability_off_target": float(prob[~positive].mean())
            if (~positive).any()
            else None,
        }
        row["step_seconds"] = time.perf_counter() - started
        self.training_seconds += row["step_seconds"]
        return row

    def select(self) -> None:
        """Freeze evidence at the same update as the chosen task state."""
        self.selected_gradients = self.active_gradients.copy()
        self.selected_updates = self.updated.copy()

    def record_validation(
        self,
        step: int,
        evaluation: dict[str, Any],
        candidate_sizes: dict[str, int],
        *,
        replay: bool,
    ) -> None:
        rows = evaluation["observations"]
        voxels = sum(candidate_sizes[r["candidate_id"]] for r in rows)
        self.validations.append(
            {
                "step": step,
                "selected_replay": replay,
                "patient_macro_dice": evaluation["patient_macro"]["dice"],
                "predicted_foreground_fraction_pooled": sum(r["predicted_voxels"] for r in rows)
                / voxels,
                "target_foreground_fraction_pooled": sum(r["reference_voxels"] for r in rows)
                / voxels,
                "empty_predictions": evaluation["counts"]["empty_predictions"],
                "reader_targets": len(rows),
            }
        )

    def finish(
        self, model: FrozenRepresentation, selected_logits: str, steps: int
    ) -> dict[str, Any]:
        replay = output_fingerprint(model)
        if replay != selected_logits:
            raise RuntimeError("Selected task state failed exact label-free logit replay.")
        selected = {name: state_fingerprint(m) for name, m in self.groups.items()}
        changed = {name: final != self.initial[name] for name, final in selected.items()}
        verified = set(self.groups) == self.selected_gradients == self.selected_updates and all(
            changed.values()
        )
        return {
            "adaptation_status": "PASS" if verified else "FAIL",
            "initial_logits_sha256": self.initial_logits,
            "selected_logits_sha256": selected_logits,
            "reloaded_logits_sha256": replay,
            "initial_group_sha256": self.initial,
            "selected_group_sha256": selected,
            "selected_group_changed": changed,
            "selected_nonzero_gradient_groups": sorted(self.selected_gradients),
            "selected_nonzero_update_groups": sorted(self.selected_updates),
            "nonzero_gradient_groups": sorted(self.active_gradients),
            "nonzero_update_groups": sorted(self.updated),
            "first_training_feature_statistics": self.features,
            "validation_trajectory": self.validations,
            "training_steps_seconds": self.training_seconds,
            "mean_step_seconds": self.training_seconds / steps,
            "validation_seconds": self.validation_seconds,
        }
