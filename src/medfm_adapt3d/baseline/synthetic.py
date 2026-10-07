"""Deterministic, explicitly synthetic fixtures through the real preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk
import yaml

from medfm_adapt3d.baseline.candidates import save_candidate, write_candidate_index
from medfm_adapt3d.preprocessing import CandidatePreprocessor, load_preprocessing_protocol


def create_synthetic_candidates(
    output: Path,
    *,
    patients: int = 34,
    preprocessing_path: Path = Path("configs/preprocessing/lidc_candidate.yaml"),
) -> Path:
    """Small CPU fixtures, not LIDC data or evidence of segmentation performance.

    Using Commit-03 preprocessing with an explicitly saved 16 mm crop solely for
    CI speed. Real-data preparation uses the existing 64 mm protocol unchanged.
    """
    output.mkdir(parents=True, exist_ok=False)
    payload = yaml.safe_load(preprocessing_path.read_text(encoding="utf-8"))
    payload["spatial_preprocessing"]["candidate_crop_extent_mm"] = [16.0] * 3
    protocol_path = output / "synthetic_preprocessing.yaml"
    with protocol_path.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False)
    protocol = load_preprocessing_protocol(protocol_path)
    preprocessor = CandidatePreprocessor(protocol.preprocessing, protocol.geometry_qc)
    z, y, x = np.indices((25, 25, 25))
    distance = (x - 12) ** 2 + (y - 12) ** 2 + (z - 12) ** 2
    records = []
    for patient in range(patients):
        for reader in range(1, 2 + patient % 2):
            target = (distance <= (2.5 + 0.3 * reader) ** 2).astype(np.uint8)
            intensity = np.where(target, 100.0 + patient, -800.0).astype(np.float32)
            image, mask = sitk.GetImageFromArray(intensity), sitk.GetImageFromArray(target)
            for volume in (image, mask):
                volume.SetOrigin((-12.0, -12.0, -12.0))
            candidate = preprocessor(image, candidate_center_mm_lps=(0.0, 0.0, 0.0), mask=mask)
            records.append(
                save_candidate(
                    output,
                    candidate,
                    patient_id=f"synthetic-{patient:03d}",
                    study_uid=f"synthetic-study-{patient}",
                    series_uid=f"synthetic-series-{patient}",
                    reader_slot=reader,
                    annotation_id=f"synthetic-target-{reader}",
                )
            )
    return write_candidate_index(
        output,
        records,
        preprocessing_path=protocol_path,
        source_audit={"evidence_source": "synthetic sphere phantom"},
        synthetic=True,
    )
