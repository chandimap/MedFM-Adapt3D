"""Explicit, auditable control over which PyTorch parameters are adapted."""

from __future__ import annotations

from collections.abc import Sequence

from torch import nn


def freeze_all(
    model: nn.Module,
) -> None:
    """Freezing every parameter in ``model``."""

    for parameter in model.parameters():
        parameter.requires_grad = False


def enable_trainable_prefixes(
    model: nn.Module,
    prefixes: Sequence[str],
) -> tuple[str, ...]:
    """Freezing the model, then unfreezing parameters matching explicit prefixes.

    Returning matched parameter names makes the adaptation policy inspectable
    and testable.

    The function fails if nothing matches, preventing silent
    experiments believing a model component is trainable when
    it is not.


    """

    normalized = tuple(
        prefix.strip()
        for prefix in prefixes
        if prefix.strip()
    )

    if not normalized:
        raise ValueError(
            "At least one non-empty parameter-name prefix is required."
        )

    freeze_all(model)

    matched: list[str] = []

    for name, parameter in model.named_parameters():
        matches = any(
            name == prefix
            or name.startswith(f"{prefix}.")
            for prefix in normalized
        )

        if matches:
            parameter.requires_grad = True
            matched.append(name)

    if not matched:
        raise ValueError(
            "Adaptation policy matched no parameters. "
            "Refusing to continue because the intended "
            "trainable subset may be incorrectly specified."
        )

    return tuple(matched)