from __future__ import annotations

import pytest

from medfm_adapt3d.data.schema import (
    CaseRecord,
    DatasetName,
)
from medfm_adapt3d.data.splits import (
    ProtocolViolation,
    build_nested_support_sets,
    validate_disjoint_patient_partitions,
)


def _cases(
    count: int,
) -> list[CaseRecord]:
    return [
        CaseRecord(
            case_id=f"case_{index:03d}",
            patient_id=f"patient_{index:03d}",
            dataset=DatasetName.LIDC_IDRI,
        )
        for index in range(count)
    ]


def test_nested_support_sets_are_reproducible_and_monotonic() -> None:
    train_pool = _cases(20)

    first = build_nested_support_sets(
        train_pool,
        (1, 5, 10, 20),
        seed=17,
    )

    second = build_nested_support_sets(
        train_pool,
        (1, 5, 10, 20),
        seed=17,
    )

    assert first == second

    assert (
        set(first[1].patient_ids)
        <= set(first[5].patient_ids)
    )

    assert (
        set(first[5].patient_ids)
        <= set(first[10].patient_ids)
    )

    assert (
        set(first[10].patient_ids)
        <= set(first[20].patient_ids)
    )


def test_k_counts_patients_not_files() -> None:
    train_pool = [
        CaseRecord(
            "case_a1",
            "patient_a",
            DatasetName.LIDC_IDRI,
        ),
        CaseRecord(
            "case_a2",
            "patient_a",
            DatasetName.LIDC_IDRI,
        ),
        CaseRecord(
            "case_b1",
            "patient_b",
            DatasetName.LIDC_IDRI,
        ),
    ]

    support = build_nested_support_sets(
        train_pool,
        (1, 2),
        seed=3,
    )

    assert len(
        support[1].patient_ids
    ) == 1

    assert len(
        support[2].patient_ids
    ) == 2

    assert set(
        support[2].case_ids
    ) == {
        "case_a1",
        "case_a2",
        "case_b1",
    }


def test_patient_leakage_is_rejected() -> None:
    train_pool = [
        CaseRecord(
            "train",
            "patient_001",
            DatasetName.LIDC_IDRI,
        )
    ]

    validation = [
        CaseRecord(
            "val",
            "patient_002",
            DatasetName.LIDC_IDRI,
        )
    ]

    test = [
        CaseRecord(
            "test",
            "patient_001",
            DatasetName.LIDC_IDRI,
        )
    ]

    with pytest.raises(
        ProtocolViolation,
        match="Patient leakage",
    ):
        validate_disjoint_patient_partitions(
            train_pool,
            validation,
            test,
        )


def test_impossible_k_is_rejected() -> None:
    with pytest.raises(
        ProtocolViolation,
        match="only 3 unique patients",
    ):
        build_nested_support_sets(
            _cases(3),
            (5,),
            seed=17,
        )