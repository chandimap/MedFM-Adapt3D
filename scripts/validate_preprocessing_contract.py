"""Run a Data-Free Scientific Check of the CT Preprocessing Contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from medfm_adapt3d.preprocessing import (
    CandidatePreprocessor,
    load_preprocessing_protocol,
)


def _synthetic_tiny_nodule() -> tuple[sitk.Image, sitk.Image, tuple[float, float, float]]:
    size_xyz = (80, 80, 32)
    spacing_xyz = (0.8, 0.8, 2.0)
    origin_xyz = (-31.6, -31.6, -31.0)
    centre_lps = (0.0, 0.0, 0.0)

    z, y, x = np.indices((size_xyz[2], size_xyz[1], size_xyz[0]))
    physical_x = origin_xyz[0] + x * spacing_xyz[0]
    physical_y = origin_xyz[1] + y * spacing_xyz[1]
    physical_z = origin_xyz[2] + z * spacing_xyz[2]
    squared_distance = (
        (physical_x - centre_lps[0]) ** 2
        + (physical_y - centre_lps[1]) ** 2
        + (physical_z - centre_lps[2]) ** 2
    )
    mask_array = (squared_distance <= 1.5**2).astype(np.uint8)
    image_array = np.full(mask_array.shape, -800.0, dtype=np.float32)
    image_array[mask_array == 1] = 100.0

    image = sitk.GetImageFromArray(image_array)
    mask = sitk.GetImageFromArray(mask_array)
    for volume in (image, mask):
        volume.SetSpacing(spacing_xyz)
        volume.SetOrigin(origin_xyz)

    return image, mask, centre_lps


def main() -> None:
    protocol = load_preprocessing_protocol(
        Path("configs/preprocessing/lidc_candidate.yaml")
    )
    image, mask, centre = _synthetic_tiny_nodule()
    result = CandidatePreprocessor(
        protocol.preprocessing,
        protocol.geometry_qc,
    )(
        image,
        candidate_center_mm_lps=centre,
        mask=mask,
    )

    audit = result.trace.mask_geometry
    if audit is None or not audit.passed:
        raise RuntimeError("Synthetic nodule did not pass geometry-preservation QC.")

    print("MedFM-Adapt3D preprocessing contract: PASS")
    print(f"Output tensor shape [C,D,H,W]: {tuple(result.image.shape)}")
    print(f"Resampled mask volume ratio: {audit.resampled_volume_ratio:.4f}")
    print(f"Tiny-nodule tolerance applied: {audit.tiny_nodule_rule_applied}")
    print(f"Mask centroid shift: {audit.centroid_shift_mm:.4f} mm")
    print(f"Crop mask retention: {audit.crop_volume_retention:.4f}")
    print("Evidence source: deterministic synthetic 3 mm nodule phantom")
    print("No clinical-performance claim is made.")


if __name__ == "__main__":
    main()
