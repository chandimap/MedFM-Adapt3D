"""Shared synthetic fixtures; no private medical data or performance assertions."""

from pathlib import Path

import pytest
import torch

from medfm_adapt3d.baseline.candidates import CandidateIndex
from medfm_adapt3d.baseline.config import BaselineConfig, load_baseline_config
from medfm_adapt3d.baseline.protocol import LockedProtocol, create_partition
from medfm_adapt3d.baseline.synthetic import create_synthetic_candidates


@pytest.fixture(scope="session")
def baseline_config() -> BaselineConfig:
    torch.set_num_threads(2)
    return load_baseline_config(Path("configs/baseline/random_segresnet.yaml"))


@pytest.fixture(scope="session")
def candidate_index(tmp_path_factory: pytest.TempPathFactory) -> CandidateIndex:
    root = tmp_path_factory.mktemp("baseline") / "candidates"
    return CandidateIndex(create_synthetic_candidates(root, patients=34))


@pytest.fixture
def locked_protocol(
    candidate_index: CandidateIndex,
    baseline_config: BaselineConfig,
    tmp_path: Path,
) -> LockedProtocol:
    path = tmp_path / "partitions.json"
    create_partition(candidate_index, baseline_config, path)
    return LockedProtocol(candidate_index, baseline_config, path)
