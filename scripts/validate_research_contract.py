"""Run a fast, data-free validation of the Commit-1 research contract."""

from __future__ import annotations

from monai.networks.nets import SegResNet

from medfm_adapt3d.adaptation.parameter_audit import (
    audit_parameter_budget,
)
from medfm_adapt3d.adaptation.policy import (
    enable_trainable_prefixes,
)
from medfm_adapt3d.data.provenance import (
    are_independent_for_external_evaluation,
)
from medfm_adapt3d.data.schema import (
    CaseRecord,
    DatasetName,
)
from medfm_adapt3d.data.splits import (
    build_nested_support_sets,
)
from medfm_adapt3d.engineering.reproducibility import (
    configure_reproducibility,
)


def main() -> None:
    configure_reproducibility(
        seed=17,
        use_deterministic_algorithms=True,
    )

    synthetic_manifest = [
        CaseRecord(
            case_id=f"synthetic_case_{index:03d}",
            patient_id=f"synthetic_patient_{index:03d}",
            dataset=DatasetName.LIDC_IDRI,
        )
        for index in range(20)
    ]

    support_sets = build_nested_support_sets(
        synthetic_manifest,
        k_values=(
            1,
            5,
            10,
            20,
        ),
        seed=17,
    )

    reference_model = SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        init_filters=8,
        upsample_mode="deconv",
    )

    full_budget = audit_parameter_budget(
        reference_model
    )

    enable_trainable_prefixes(
        reference_model,
        prefixes=("conv_final",),
    )

    head_budget = audit_parameter_budget(
        reference_model
    )

    print(
        "MedFM-Adapt3D Commit-1 research contract: PASS"
    )

    print(
        f"Nested K-shot budgets: {tuple(support_sets)}"
    )

    independent = (
        are_independent_for_external_evaluation(
            DatasetName.LIDC_IDRI,
            DatasetName.LUNA16,
        )
    )

    print(
        "LIDC-IDRI -> LUNA16 independent "
        f"external evaluation: {independent}"
    )

    print(
        "Reference-model trainable parameters (full): "
        f"{full_budget.trainable_parameters:,}"
    )

    print(
        "Reference-model trainable parameters (head-only): "
        f"{head_budget.trainable_parameters:,} "
        f"({head_budget.trainable_percentage:.4f}%)"
    )

    print(
        "Synthetic IDs above are protocol fixtures, "
        "not experimental data or results."
    )


if __name__ == "__main__":
    main()