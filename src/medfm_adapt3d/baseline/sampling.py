"""Balanced Patient Exposure while Rotating all available Reader Observations."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence

import torch

from medfm_adapt3d.baseline.candidates import CandidateRecord


class PatientBalancedSampler:
    """Each shuffled patient cycle visits every patient exactly once.

    Each patient's separately shuffled observation cycle visits all eligible
    reader targets before repeating. Annotation-rich patients get more diverse
    targets, not more optimizer influence. Worker-free loading is intentional.
    """

    def __init__(self, records: Sequence[CandidateRecord], loader_seed: int) -> None:
        self.generator = torch.Generator().manual_seed(loader_seed)
        grouped: dict[str, list[str]] = defaultdict(list)
        for record in records:
            grouped[record.patient_id].append(record.candidate_id)
        if not grouped:
            raise ValueError("Patient-balanced sampling requires non-empty support.")
        self.targets = {p: sorted(ids) for p, ids in sorted(grouped.items())}
        self._patient_cycle: list[str] = []
        self._target_cycles: dict[str, list[str]] = {p: [] for p in self.targets}
        self.patient_exposure: Counter[str] = Counter({p: 0 for p in self.targets})
        self.target_exposure: Counter[str] = Counter({r.candidate_id: 0 for r in records})

    def _shuffled(self, values: Sequence[str]) -> list[str]:
        permutation = torch.randperm(len(values), generator=self.generator)
        return [values[int(index.item())] for index in permutation]

    def next_id(self) -> str:
        if not self._patient_cycle:
            self._patient_cycle = self._shuffled(list(self.targets))
        patient = self._patient_cycle.pop()
        if not self._target_cycles[patient]:
            self._target_cycles[patient] = self._shuffled(self.targets[patient])
        target = self._target_cycles[patient].pop()
        self.patient_exposure[patient] += 1
        self.target_exposure[target] += 1
        return target


def augment_pair(
    image: torch.Tensor,
    mask: torch.Tensor,
    *,
    generator: torch.Generator,
    probability: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Conservative identity augmentation preserves normalized anatomical orientation.

    No anterior/posterior, superior/inferior or left/right reflection is assumed
    plausible for this reference. The separate augmentation RNG is retained but
    intentionally consumes no draws. Later methods receive the same opportunity.
    """
    if probability != 0:
        raise ValueError("The primary reference permits no spatial reflections.")
    if image.shape != mask.shape:
        raise ValueError("Paired CT and target must have identical geometry.")
    return image, mask
