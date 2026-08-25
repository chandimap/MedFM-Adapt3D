"""Typed data structures for dataset provenance and patient-level experiments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class DatasetName(StrEnum):
    """Target datasets for the MedFM-Adapt3D research programme."""

    LIDC_IDRI = "LIDC-IDRI"
    LUNA16 = "LUNA16"
    LNDB = "LNDb"
    LUNA25 = "LUNA25"


@dataclass(frozen=True, slots=True)
class CaseRecord:
    """One imaging case while keeping patient identity explicit for leakage control.

    A patient can own more than one imaging case. Splitting is therefore performed
    on ``patient_id`` rather than ``case_id``.
    """

    case_id: str
    patient_id: str
    dataset: DatasetName

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id must be a non-empty string.")

        if not self.patient_id.strip():
            raise ValueError("patient_id must be a non-empty string.")


@dataclass(frozen=True, slots=True)
class SupportSet:
    """A reproducible K-shot support set defined in units of patients."""

    k: int
    patient_ids: tuple[str, ...]
    case_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError("k must be positive.")

        if len(self.patient_ids) != self.k:
            raise ValueError("k must equal the number of selected patients.")