"""Parameter-budget accounting for controlled adaptation experiments."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True, slots=True)
class ParameterBudget:
    """Absolute and relative parameter counts for one adaptation strategy."""

    total_parameters: int
    trainable_parameters: int

    @property
    def frozen_parameters(self) -> int:
        return (
            self.total_parameters
            - self.trainable_parameters
        )

    @property
    def trainable_fraction(self) -> float:
        if self.total_parameters == 0:
            return 0.0

        return (
            self.trainable_parameters
            / self.total_parameters
        )

    @property
    def trainable_percentage(self) -> float:
        return 100.0 * self.trainable_fraction


def audit_parameter_budget(
    model: nn.Module,
) -> ParameterBudget:
    """Counting total and trainable parameters directly from a PyTorch module."""

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return ParameterBudget(
        total_parameters=total,
        trainable_parameters=trainable,
    )