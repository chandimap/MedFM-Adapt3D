"""Reproducibility controls and machine-readable runtime provenance."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import monai
import torch
from monai.utils import set_determinism


@dataclass(frozen=True, slots=True)
class RuntimeProvenance:
    """Minimal environment information required to interpret an experiment run."""

    python_version: str
    pytorch_version: str
    monai_version: str
    platform: str
    cuda_available: bool
    cuda_version: str | None
    cudnn_version: int | None
    device_name: str | None
    git_commit: str | None


def configure_reproducibility(
    seed: int,
    *,
    use_deterministic_algorithms: bool = True,
) -> None:
    """Configuring MONAI/PyTorch reproducibility controls.

    Determinism is an experimental setting.
    """

    if seed < 0:
        raise ValueError(
            "seed must be non-negative."
        )

    set_determinism(
        seed=seed,
        use_deterministic_algorithms=use_deterministic_algorithms,
    )


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )

    except (
        FileNotFoundError,
        subprocess.SubprocessError,
    ):
        return None

    commit = result.stdout.strip()

    return commit or None


def collect_runtime_provenance() -> RuntimeProvenance:
    """Collecting software, hardware, and version-control provenance."""

    cuda_available = torch.cuda.is_available()

    device_name = (
        torch.cuda.get_device_name(0)
        if cuda_available
        else None
    )

    return RuntimeProvenance(
        python_version=sys.version.split()[0],
        pytorch_version=torch.__version__,
        monai_version=monai.__version__,
        platform=platform.platform(),
        cuda_available=cuda_available,
        cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(),
        device_name=device_name,
        git_commit=_git_commit(),
    )


def write_runtime_provenance(
    path: Path,
) -> None:
    """Writing experiment provenance as stable human-readable JSON."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = asdict(
        collect_runtime_provenance()
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )