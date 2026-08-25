"""Dataset-lineage safeguards for scientifically valid external evaluation."""

from __future__ import annotations

from medfm_adapt3d.data.schema import DatasetName
from medfm_adapt3d.data.splits import ProtocolViolation

# LUNA16 is a curated benchmark derived from LIDC-IDRI.
# Therefore the two datasets are non-independent for external-validation claims.
_NON_INDEPENDENT_DATASET_PAIRS = frozenset(
    {
        frozenset(
            {
                DatasetName.LIDC_IDRI,
                DatasetName.LUNA16,
            }
        ),
    }
)


def are_independent_for_external_evaluation(
    development_dataset: DatasetName,
    evaluation_dataset: DatasetName,
) -> bool:
    """Returning whether a dataset pair is eligible for an independence claim."""

    if development_dataset == evaluation_dataset:
        return False

    pair = frozenset(
        {
            development_dataset,
            evaluation_dataset,
        }
    )

    return pair not in _NON_INDEPENDENT_DATASET_PAIRS


def require_external_independence(
    development_dataset: DatasetName,
    evaluation_dataset: DatasetName,
) -> None:
    """Rejecting a scientifically invalid claim of independent external evaluation."""

    if not are_independent_for_external_evaluation(
        development_dataset,
        evaluation_dataset,
    ):
        raise ProtocolViolation(
            f"{evaluation_dataset} must not be presented as an independent "
            f"external evaluation cohort for {development_dataset}. "
            "Record the dataset lineage and use a genuinely independent "
            "cohort for external-generalisation claims."
        )