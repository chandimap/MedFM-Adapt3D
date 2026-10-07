"""Validation-only Selection and a Stateful Barrier around Held-out Test Targets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from medfm_adapt3d.baseline.artifacts import (
    file_hash,
    fingerprint,
    implementation_fingerprint,
    read_sealed,
)
from medfm_adapt3d.baseline.config import Optimization, Phase
from medfm_adapt3d.baseline.protocol import LockedProtocol


@dataclass
class CheckpointSelector:
    budget: Optimization
    best_value: float = -1.0
    best_step: int = 0
    meaningful_value: float = -1.0
    stale_validations: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    def observe(self, *, step: int, patient_macro_dice: float, source: str) -> bool:
        """Strictly best Dice selects the checkpoint; min_delta controls stopping only."""
        if source != "validation":
            raise ValueError(
                "Only validation may select a checkpoint; test/training scores rejected."
            )
        if not 0 <= patient_macro_dice <= 1 or step <= 0:
            raise ValueError("Validation Dice and step are invalid.")
        if self.history and step <= self.history[-1]["step"]:
            raise ValueError("Validation steps must increase monotonically.")
        improved = patient_macro_dice > self.best_value
        if improved:
            self.best_value, self.best_step = patient_macro_dice, step
        if patient_macro_dice > self.meaningful_value + self.budget.min_delta:
            self.meaningful_value, self.stale_validations = patient_macro_dice, 0
        else:
            self.stale_validations += 1
        self.history.append(
            {
                "step": step,
                "patient_macro_dice": patient_macro_dice,
                "selected": improved,
                "source": "validation",
            }
        )
        return improved

    def should_stop(self, step: int) -> bool:
        return step >= self.budget.min_steps and self.stale_validations >= self.budget.patience


def require_optimization_adequacy(diagnostics: dict[str, Any]) -> None:
    """Rejecting inadequate train/validation evidence BEFORE any definitive test access.

    The prespecified patience/min_delta rule defines plateau; this is an
    engineering release gate, not evidence of a globally optimal solution.
    """
    required = (
        "checkpoint_selection_complete",
        "final_validation_complete",
        "minimum_steps_satisfied",
        "numerically_finite",
    )
    if any(diagnostics.get(name) is not True for name in required):
        raise RuntimeError("Optimization adequacy requires complete, finite minimum-step evidence.")
    if diagnostics.get("budget_exhausted_without_patience_plateau") is not False:
        raise RuntimeError(
            "Definitive run exhausted the prespecified optimization budget while validation "
            "remained meaningfully improving. Held-out test access is blocked because "
            "baseline optimization adequacy has not been established."
        )
    if any(
        diagnostics.get(name) is not False
        for name in (
            "all_validation_predictions_empty",
            "all_validation_predictions_full",
            "zero_validation_overlap",
        )
    ):
        raise RuntimeError("Collapsed validation predictions block optimization adequacy/test.")
    if (
        any(
            diagnostics.get(name) != 0
            for name in ("unvisited_training_patients", "unvisited_training_targets")
        )
        or diagnostics.get("patient_exposure_balanced") is not True
    ):
        raise RuntimeError("Incomplete/unbalanced patient or reader exposure blocks test access.")


class RunAccess:
    """Support-only training; definitive targets/QC require validation AND adequacy.

    This enforces the project execution path, not malicious Python modifications
    or operating-system access by the researcher. Calibration never unlocks test.
    """

    def __init__(self, protocol: LockedProtocol, k: int, phase: Phase) -> None:
        self._protocol = protocol
        self.phase = phase
        self.allowed = {
            "train": protocol.support_ids(k),
            "validation": protocol.partition_ids("validation"),
            "test": protocol.partition_ids("test"),
        }
        self.state = "training"
        self.lock_payload: dict[str, Any] | None = None
        self.events: list[dict[str, Any]] = []
        self.read_counts = {name: 0 for name in self.allowed}
        self._final_validation_seen: set[str] = set()
        self._test_seen: set[str] = set()
        self._checkpoint: Path | None = None
        self._adequacy_path: Path | None = None
        self._adequacy_sha256: str | None = None

    def read(self, partition: str, candidate_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        if partition not in self.allowed or candidate_id not in self.allowed[partition]:
            raise ValueError("Candidate access outside the authorized patient partition/support.")
        if partition == "test" and (self.phase == "calibration" or self.state != "test_open"):
            raise RuntimeError(
                "Test arrays/QC are sealed until validation and optimization adequacy."
            )
        if partition == "train" and self.state != "training":
            raise RuntimeError("Training targets cannot be used after the checkpoint is locked.")
        if partition == "validation" and self.state == "locked":
            if candidate_id in self._final_validation_seen:
                raise RuntimeError("Final validation must visit each observation once.")
            self._final_validation_seen.add(candidate_id)
        if partition == "test":
            if candidate_id in self._test_seen:
                raise RuntimeError("Held-out test observations cannot be evaluated repeatedly.")
            self._test_seen.add(candidate_id)
        self.read_counts[partition] += 1
        return self._protocol.index.load(candidate_id)

    def lock(self, lock_path: Path, checkpoint: Path) -> None:
        if self.state != "training":
            raise RuntimeError("Checkpoint can be locked only once.")
        payload = read_sealed(lock_path)
        if payload["checkpoint_sha256"] != file_hash(checkpoint):
            raise ValueError("Selected checkpoint integrity failure.")
        if payload["partition_sha256"] != self._protocol.sha256 or (
            payload["candidate_index_sha256"] != self._protocol.index.sha256
        ):
            raise ValueError("Checkpoint lock belongs to a different patient protocol.")
        if payload["selection_source"] != "validation":
            raise ValueError("Checkpoint lock must attest validation-only selection.")
        self._verify_protocol_binding(payload)
        self.lock_payload = payload
        self._checkpoint = checkpoint
        self.state = "locked"
        self.events.append({"event": "checkpoint_locked", "test_label_reads": 0})

    def final_validation_completed(self) -> None:
        if self.state != "locked" or self._final_validation_seen != set(self.allowed["validation"]):
            raise RuntimeError("Final validation must follow the checkpoint lock.")
        self.state = "validated"
        self.events.append({"event": "final_validation_completed", "test_label_reads": 0})

    def _verify_protocol_binding(self, payload: dict[str, Any]) -> None:
        """Reject source, settings or on-disk cohort drift before releasing any target."""
        expected = {
            "config_sha256": fingerprint(self._protocol.config.to_dict()),
            "implementation_sha256": implementation_fingerprint(),
            "partition_sha256": self._protocol.sha256,
            "candidate_index_sha256": self._protocol.index.sha256,
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise ValueError("Checkpoint/gate recipe, source or cohort no longer matches.")
        if (
            fingerprint(read_sealed(self._protocol.path)) != self._protocol.sha256
            or fingerprint(read_sealed(self._protocol.index.path)) != self._protocol.index.sha256
        ):
            raise ValueError("On-disk partition/candidate index changed before test access.")

    def approve_optimization(self, gate_path: Path) -> None:
        """Transition a definitive run to test-ready using sealed pre-test evidence."""
        if self.phase != "definitive" or self.state != "validated":
            raise RuntimeError("Optimization approval requires definitive locked final validation.")
        gate = read_sealed(gate_path)
        self._verify_protocol_binding(gate)
        require_optimization_adequacy(gate["diagnostics"])
        if (
            self.lock_payload is None
            or gate["checkpoint_sha256"] != self.lock_payload["checkpoint_sha256"]
            or gate["selected_step"] != self.lock_payload["selected_step"]
            or not self._protocol.config.optimization.min_steps
            <= gate["actual_steps"]
            <= self._protocol.config.optimization.max_steps
            or gate["test_label_reads"] != 0
            or self.read_counts["test"] != 0
            or gate["status"] != "passed"
        ):
            raise ValueError(
                "Optimization gate disagrees with locked weights, steps or test access."
            )
        self._adequacy_path, self._adequacy_sha256 = gate_path, file_hash(gate_path)
        self.state = "test_ready"
        self.events.append({"event": "optimization_adequacy_passed", "test_label_reads": 0})

    def open_test(self) -> None:
        required_state = "test_ready" if self.phase == "definitive" else "validated"
        if self.phase == "calibration" or self.state != required_state:
            raise RuntimeError(
                "Test evaluation requires locked weights, completed validation and "
                "definitive optimization adequacy."
            )
        if self.phase == "smoke" and not self._protocol.index.payload["synthetic"]:
            raise RuntimeError("Real-data smoke cannot release held-out arrays or target QC.")
        if (
            self._checkpoint is None
            or self.lock_payload is None
            or (file_hash(self._checkpoint) != self.lock_payload["checkpoint_sha256"])
        ):
            raise ValueError("Checkpoint changed before test access.")
        self._verify_protocol_binding(self.lock_payload)
        if self.phase == "definitive" and (
            self._adequacy_path is None or file_hash(self._adequacy_path) != self._adequacy_sha256
        ):
            raise ValueError("Optimization adequacy evidence changed before test access.")
        self.state = "test_open"
        self.events.append({"event": "test_evaluation_started", "test_label_reads": 0})

    def finish_test(self) -> None:
        if self.state != "test_open" or self._test_seen != set(self.allowed["test"]):
            raise RuntimeError(
                "Final test evaluation must read each held-out observation exactly once."
            )
        self.state = "complete"
        self.events.append(
            {"event": "test_evaluation_completed", "test_label_reads": self.read_counts["test"]}
        )
