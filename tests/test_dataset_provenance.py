from __future__ import annotations

import pytest

from medfm_adapt3d.data.provenance import (
    are_independent_for_external_evaluation,
    require_external_independence,
)
from medfm_adapt3d.data.schema import (
    DatasetName,
)
from medfm_adapt3d.data.splits import (
    ProtocolViolation,
)


def test_luna16_is_not_independent_from_lidc_idri() -> None:
    assert not are_independent_for_external_evaluation(
        DatasetName.LIDC_IDRI,
        DatasetName.LUNA16,
    )


def test_invalid_external_claim_is_rejected() -> None:
    with pytest.raises(
        ProtocolViolation,
        match="must not be presented",
    ):
        require_external_independence(
            DatasetName.LIDC_IDRI,
            DatasetName.LUNA16,
        )