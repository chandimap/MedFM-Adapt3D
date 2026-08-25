"""Leakage-safe and reproducible patient-level few-shot split utilities."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import torch

from medfm_adapt3d.data.schema import CaseRecord, SupportSet


class ProtocolViolation(ValueError):
    """Raised when an experimental-design invariant is violated."""


def _patients(cases: Sequence[CaseRecord]) -> set[str]:
    return {case.patient_id for case in cases}


def validate_disjoint_patient_partitions(
    train_pool: Sequence[CaseRecord],
    validation: Sequence[CaseRecord],
    test: Sequence[CaseRecord],
) -> None:
    """Rejecting patient overlap across train, validation, and test partitions."""

    partitions: Mapping[str, set[str]] = {
        "train_pool": _patients(train_pool),
        "validation": _patients(validation),
        "test": _patients(test),
    }

    names = tuple(partitions)

    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            overlap = partitions[left_name] & partitions[right_name]

            if overlap:
                examples = ", ".join(sorted(overlap)[:5])

                raise ProtocolViolation(
                    "Patient leakage detected between "
                    f"{left_name} and {right_name}: {examples}"
                )


def build_nested_support_sets(
    train_pool: Sequence[CaseRecord],
    k_values: Sequence[int],
    seed: int,
) -> dict[int, SupportSet]:
    """Building deterministic nested K-shot cohorts using a local PyTorch RNG.

    ``K`` is defined as the number of unique patients, not the number of files.
    If a selected patient owns multiple cases, all of those cases remain together.

    A local ``torch.Generator`` is used so cohort construction is reproducible
    without mutating PyTorch's global random-number-generator state.
    """

    if not train_pool:
        raise ProtocolViolation(
            "train_pool must contain at least one case."
        )

    if not k_values:
        raise ProtocolViolation(
            "k_values must contain at least one K-shot budget."
        )

    if any(k <= 0 for k in k_values):
        raise ProtocolViolation(
            "Every K-shot budget must be a positive integer."
        )

    patient_to_cases: dict[str, list[str]] = defaultdict(list)

    for case in train_pool:
        patient_to_cases[case.patient_id].append(case.case_id)

    patient_ids = sorted(patient_to_cases)

    maximum_k = max(k_values)

    if maximum_k > len(patient_ids):
        raise ProtocolViolation(
            f"Requested K={maximum_k}, but only {len(patient_ids)} "
            "unique patients are available in the adaptation pool."
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    permutation = torch.randperm(
        len(patient_ids),
        generator=generator,
    ).tolist()

    ordered_patients = [
        patient_ids[index]
        for index in permutation
    ]

    support_sets: dict[int, SupportSet] = {}

    for k in sorted(set(k_values)):
        selected_patients = tuple(
            ordered_patients[:k]
        )

        selected_cases = tuple(
            case_id
            for patient_id in selected_patients
            for case_id in sorted(patient_to_cases[patient_id])
        )

        support_sets[k] = SupportSet(
            k=k,
            patient_ids=selected_patients,
            case_ids=selected_cases,
        )

    return support_sets