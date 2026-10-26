"""Checkpoint Identity and an Explicit Limit on Claims from Unresolved Provenance."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from medfm_adapt3d.baseline.artifacts import fingerprint

MODEL_ID = "suprem-segresnet-2100"
CHECKPOINT_REVISION = "46ab78c25b07ace573177ebffd02214f670c7e4f"
CHECKPOINT_SHA256 = "2db81dc05cd9ea7234ca75e921e53e32b8716dc4cba88a6710742bfc282589a3"
CHECKPOINT_BYTES = 56500623


@dataclass(frozen=True, slots=True)
class ModelRecord:
    schema_version: str
    model_id: str
    upstream_repository: str
    upstream_revision: str
    checkpoint_repository: str
    checkpoint_revision: str
    checkpoint_filename: str
    checkpoint_sha256: str
    checkpoint_size_bytes: int
    overlap_status: str
    license_status: str
    evidence: tuple[dict[str, str], ...]

    def __post_init__(self) -> None:
        expected = (
            "1.0",
            MODEL_ID,
            "MrGiovanni/SuPreM",
            CHECKPOINT_REVISION,
            "supervised_suprem_segresnet_2100.pth",
            CHECKPOINT_SHA256,
            CHECKPOINT_BYTES,
        )
        actual = (
            self.schema_version,
            self.model_id,
            self.checkpoint_repository,
            self.checkpoint_revision,
            self.checkpoint_filename,
            self.checkpoint_sha256,
            self.checkpoint_size_bytes,
        )
        if actual != expected:
            raise ValueError("Unsupported checkpoint identity; this adapter supports one release.")
        if self.upstream_repository != "https://github.com/MrGiovanni/SuPreM":
            raise ValueError("Upstream repository identity changed.")
        if self.upstream_revision != "7583213147e33daf2031cc0a3ad30faa516f4f64":
            raise ValueError("Upstream implementation revision changed.")
        if self.overlap_status != "NOT_ESTABLISHED":
            raise ValueError("This public evidence does not establish patient-level disjointness.")
        if self.license_status != "CONFLICTING_PUBLIC_METADATA":
            raise ValueError(
                "Record both public licence notices without resolving them by assertion."
            )
        if len(self.evidence) < 4 or any(
            not isinstance(item, dict)
            or set(item) != {"url", "finding"}
            or not all(isinstance(v, str) and v.strip() for v in item.values())
            for item in self.evidence
        ):
            raise ValueError("A provenance record requires source URLs and explicit findings.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return fingerprint(self.to_dict())


def load_model_record(path: Path) -> ModelRecord:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {f.name for f in fields(ModelRecord)}:
        raise ValueError("Model provenance fields must match the supported schema exactly.")
    if not isinstance(payload["evidence"], list):
        raise ValueError("Provenance evidence must be a list of source findings.")
    payload["evidence"] = tuple(payload["evidence"])
    return ModelRecord(**payload)


def claim_gate(record: ModelRecord, requested_claim: str = "engineering") -> dict[str, Any]:
    """A checksum establishes bytes, not independence of the underlying patients."""
    if requested_claim not in ("engineering", "clean_generalization", "clinical_readiness"):
        raise ValueError("Unknown claim scope.")
    allowed = requested_claim == "engineering"
    result = {
        "model_id": record.model_id,
        "model_record_sha256": record.sha256,
        "overlap_status": record.overlap_status,
        "requested_claim": requested_claim,
        "allowed": allowed,
        "real_held_out_test_enabled": False,
        "clinical_performance_claimed": False,
        "reason": "Patient-level overlap is unresolved; only implementation evidence is supported.",
    }
    if not allowed:
        raise ValueError(result["reason"])
    return result
