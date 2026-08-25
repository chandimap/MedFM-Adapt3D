from __future__ import annotations

import pytest
from torch import nn

from medfm_adapt3d.adaptation.parameter_audit import (
    audit_parameter_budget,
)
from medfm_adapt3d.adaptation.policy import (
    enable_trainable_prefixes,
)


def test_parameter_budget_reflects_explicit_adaptation_policy() -> None:
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )

    full = audit_parameter_budget(
        model
    )

    matched = enable_trainable_prefixes(
        model,
        prefixes=("2",),
    )

    head_only = audit_parameter_budget(
        model
    )

    assert (
        full.trainable_parameters
        == full.total_parameters
    )

    assert matched == (
        "2.weight",
        "2.bias",
    )

    assert (
        head_only.trainable_parameters
        < full.trainable_parameters
    )

    assert (
        0.0
        < head_only.trainable_percentage
        < 100.0
    )


def test_bad_parameter_prefix_fails_loudly() -> None:
    model = nn.Linear(
        4,
        2,
    )

    with pytest.raises(
        ValueError,
        match="matched no parameters",
    ):
        enable_trainable_prefixes(
            model,
            prefixes=("does_not_exist",),
        )