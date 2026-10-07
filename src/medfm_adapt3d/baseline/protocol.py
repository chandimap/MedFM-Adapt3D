"""Lock Patient Membership using the Existing Nested-support Scientific Machinery."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from medfm_adapt3d.baseline.artifacts import fingerprint, read_sealed, seal, write_json
from medfm_adapt3d.baseline.candidates import CandidateIndex
from medfm_adapt3d.baseline.config import BaselineConfig
from medfm_adapt3d.data.schema import CaseRecord, DatasetName
from medfm_adapt3d.data.splits import (
    ProtocolViolation,
    build_nested_support_sets,
    validate_disjoint_patient_partitions,
)


def expected_partition(index: CandidateIndex, config: BaselineConfig) -> dict[str, Any]:
    """Fixed partition seed and support seed, independent of every model RNG."""
    patients = index.patients
    if len(patients) < 3:
        raise ProtocolViolation("At least three eligible patients are needed for three partitions.")
    n_val = max(1, int(len(patients) * config.data.validation_fraction))
    n_test = max(1, int(len(patients) * config.data.test_fraction))
    generator = torch.Generator().manual_seed(config.seeds.partition)
    order = [patients[i] for i in torch.randperm(len(patients), generator=generator).tolist()]
    groups = {
        "validation": sorted(order[:n_val]),
        "test": sorted(order[n_val : n_val + n_test]),
        "train": sorted(order[n_val + n_test :]),
    }
    cases = [CaseRecord(r.candidate_id, r.patient_id, DatasetName.LIDC_IDRI) for r in index.records]
    by_group = {name: [c for c in cases if c.patient_id in ids] for name, ids in groups.items()}
    validate_disjoint_patient_partitions(
        by_group["train"], by_group["validation"], by_group["test"]
    )
    feasible = [k for k in config.data.budgets if k <= len(groups["train"])]
    if not feasible:
        raise ProtocolViolation("No feasible support budget remains after patient partitioning.")
    supports = build_nested_support_sets(by_group["train"], feasible, config.seeds.support)
    return {
        "schema_version": "1.0",
        "candidate_index_sha256": index.sha256,
        "preprocessing_sha256": index.payload["preprocessing_sha256"],
        "partition_seed": config.seeds.partition,
        "support_seed": config.seeds.support,
        "validation_fraction": config.data.validation_fraction,
        "test_fraction": config.data.test_fraction,
        "patients": groups,
        "supports": {
            str(k): {"patient_ids": list(s.patient_ids), "case_ids": list(s.case_ids)}
            for k, s in supports.items()
        },
        "unavailable_budgets": [k for k in config.data.budgets if k not in feasible],
    }


def create_partition(index: CandidateIndex, config: BaselineConfig, path: Path) -> None:
    write_json(path, seal(expected_partition(index, config)))


class LockedProtocol:
    def __init__(self, index: CandidateIndex, config: BaselineConfig, path: Path) -> None:
        self.index = index
        self.config = config
        self.path = path.resolve()
        self.payload = read_sealed(path)
        # A checksum alone cannot reject a rewritten membership file. Recompute it.
        if self.payload != expected_partition(index, config):
            raise ProtocolViolation(
                "Tampered partition/index or changed partition/support settings."
            )
        self.sha256 = fingerprint(self.payload)

    @property
    def calibration_budgets(self) -> tuple[int, ...]:
        """Check the ONE shared recipe at the smallest and largest feasible K."""
        feasible = sorted(map(int, self.payload["supports"]))
        return tuple(sorted({feasible[0], feasible[-1]}))

    def support_ids(self, k: int) -> tuple[str, ...]:
        if type(k) is not int or k not in self.config.data.budgets:
            raise ProtocolViolation(
                "K must be one of the prespecified patient budgets (1,5,10,20)."
            )
        supports = self.payload["supports"]
        if str(k) not in supports:
            available = len(self.payload["patients"]["train"])
            raise ProtocolViolation(
                f"Requested K={k}, but only {available} eligible train patients."
            )
        return tuple(supports[str(k)]["case_ids"])

    def partition_ids(self, name: str) -> tuple[str, ...]:
        if name not in ("train", "validation", "test"):
            raise ValueError("Unknown patient partition.")
        patients = set(self.payload["patients"][name])
        return tuple(r.candidate_id for r in self.index.records if r.patient_id in patients)
