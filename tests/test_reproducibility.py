from __future__ import annotations

import torch

from medfm_adapt3d.engineering.reproducibility import (
    configure_reproducibility,
)


def test_reseeding_reproduces_pytorch_random_sequence() -> None:
    configure_reproducibility(
        seed=17
    )

    first = torch.rand(8)

    configure_reproducibility(
        seed=17
    )

    second = torch.rand(8)

    assert torch.equal(
        first,
        second,
    )