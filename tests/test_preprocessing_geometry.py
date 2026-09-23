import numpy as np
import pytest
import SimpleITK as sitk
import torch

from medfm_adapt3d.preprocessing.config import (
    GeometryPreservationCriteria,
    PreprocessingConfig,
)
from medfm_adapt3d.preprocessing.geometry import GeometryMismatchError
from medfm_adapt3d.preprocessing.pipeline import (
    CandidatePreprocessor,
    window_and_scale_ct,
)
from medfm_adapt3d.preprocessing.quality import (
    audit_mask_geometry,
    validate_binary_mask,
)


def _tiny_nodule_case() -> tuple[sitk.Image, sitk.Image, tuple[float, float, float]]:
    size_xyz = (80, 80, 32)
    spacing_xyz = (0.8, 0.8, 2.0)
    origin_xyz = (-31.6, -31.6, -31.0)
    centre = (0.0, 0.0, 0.0)

    z, y, x = np.indices((size_xyz[2], size_xyz[1], size_xyz[0]))
    squared_distance = (
        (origin_xyz[0] + x * spacing_xyz[0] - centre[0]) ** 2
        + (origin_xyz[1] + y * spacing_xyz[1] - centre[1]) ** 2
        + (origin_xyz[2] + z * spacing_xyz[2] - centre[2]) ** 2
    )
    mask_array = (squared_distance <= 1.5**2).astype(np.uint8)
    image_array = np.full(mask_array.shape, -800.0, dtype=np.float32)
    image_array[mask_array == 1] = 100.0

    image = sitk.GetImageFromArray(image_array)
    mask = sitk.GetImageFromArray(mask_array)
    for volume in (image, mask):
        volume.SetSpacing(spacing_xyz)
        volume.SetOrigin(origin_xyz)

    return image, mask, centre


def _preprocessor(*, crop_extent_mm: float = 64.0) -> CandidatePreprocessor:
    return CandidatePreprocessor(
        PreprocessingConfig(
            crop_extent_mm=(crop_extent_mm,) * 3,
        ),
        GeometryPreservationCriteria(),
    )


def test_tiny_nodule_survives_resampling_and_candidate_crop() -> None:
    image, mask, centre = _tiny_nodule_case()

    result = _preprocessor()(
        image,
        candidate_center_mm_lps=centre,
        mask=mask,
    )

    audit = result.trace.mask_geometry
    assert audit is not None
    assert audit.passed
    assert audit.resampled.voxel_count > 0
    assert audit.tiny_nodule_rule_applied
    assert (
        audit.applied_minimum_volume_ratio
        <= audit.resampled_volume_ratio
        <= audit.applied_maximum_volume_ratio
    )
    assert audit.centroid_shift_mm <= 1.50
    assert audit.crop_volume_retention == pytest.approx(1.0)
    assert result.image.shape == (1, 65, 65, 65)
    assert result.mask is not None
    assert result.mask.shape == result.image.shape
    assert result.image.dtype == torch.float32
    assert result.mask.dtype == torch.uint8
    assert set(torch.unique(result.mask).tolist()) <= {0, 1}


def test_three_mm_nodule_survives_luna16_maximum_slice_spacing() -> None:
    size_xyz = (80, 80, 24)
    spacing_xyz = (0.7, 0.7, 2.5)
    origin_xyz = tuple(
        -((length - 1) * spacing) / 2.0
        for length, spacing in zip(size_xyz, spacing_xyz, strict=True)
    )
    z, y, x = np.indices((size_xyz[2], size_xyz[1], size_xyz[0]))
    squared_distance = (
        (origin_xyz[0] + x * spacing_xyz[0]) ** 2
        + (origin_xyz[1] + y * spacing_xyz[1]) ** 2
        + (origin_xyz[2] + z * spacing_xyz[2]) ** 2
    )
    mask_array = (squared_distance <= 1.5**2).astype(np.uint8)
    image_array = np.full(mask_array.shape, -800.0, dtype=np.float32)
    image_array[mask_array == 1] = 100.0
    image = sitk.GetImageFromArray(image_array)
    mask = sitk.GetImageFromArray(mask_array)
    for volume in (image, mask):
        volume.SetSpacing(spacing_xyz)
        volume.SetOrigin(origin_xyz)

    result = _preprocessor()(
        image,
        candidate_center_mm_lps=(0.0, 0.0, 0.0),
        mask=mask,
    )

    audit = result.trace.mask_geometry
    assert audit is not None
    assert audit.passed
    assert audit.tiny_nodule_rule_applied
    assert audit.resampled.voxel_count > 0
    assert audit.centroid_shift_mm <= 1.5


def test_output_uses_axis_aligned_lps_geometry_after_flipped_input() -> None:
    array = np.full((12, 16, 20), -700.0, dtype=np.float32)
    image = sitk.GetImageFromArray(array)
    image.SetOrigin((19.0, 15.0, -5.0))
    image.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
    centre = image.TransformIndexToPhysicalPoint((10, 8, 6))

    result = _preprocessor(crop_extent_mm=8.0)(
        image,
        candidate_center_mm_lps=centre,
    )

    assert result.trace.output_geometry.direction_lps == (
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    assert result.trace.output_geometry.spacing_mm_xyz == (1.0, 1.0, 1.0)
    assert result.image.shape == (1, 9, 9, 9)


def test_candidate_crop_pads_scan_boundary_without_moving_centre() -> None:
    image = sitk.Image((12, 12, 12), sitk.sitkFloat32)
    image = image - 800.0
    centre = image.TransformIndexToPhysicalPoint((0, 0, 0))

    result = _preprocessor(crop_extent_mm=8.0)(
        image,
        candidate_center_mm_lps=centre,
    )

    assert result.image.shape == (1, 9, 9, 9)
    assert result.trace.crop_centering_error_mm == pytest.approx(0.0)
    assert float(result.image.min()) == pytest.approx(-1.0)


def test_image_mask_origin_mismatch_is_rejected_before_resampling() -> None:
    image, mask, centre = _tiny_nodule_case()
    mask.SetOrigin((0.1, mask.GetOrigin()[1], mask.GetOrigin()[2]))

    with pytest.raises(GeometryMismatchError, match="origin"):
        _preprocessor()(
            image,
            candidate_center_mm_lps=centre,
            mask=mask,
        )


def test_fractional_mask_labels_are_rejected() -> None:
    mask = sitk.Image((8, 8, 8), sitk.sitkFloat32)
    mask[3, 3, 3] = 0.5

    with pytest.raises(ValueError, match="binary"):
        validate_binary_mask(mask)


def test_larger_mask_uses_tighter_standard_volume_bounds() -> None:
    array = np.zeros((12, 12, 12), dtype=np.uint8)
    array[3:9, 3:9, 3:9] = 1
    mask = sitk.GetImageFromArray(array)
    criteria = GeometryPreservationCriteria()

    audit = audit_mask_geometry(
        mask,
        mask,
        mask,
        criteria=criteria,
    )

    assert not audit.tiny_nodule_rule_applied
    assert audit.applied_minimum_volume_ratio == 0.65
    assert audit.applied_maximum_volume_ratio == 1.50
    assert audit.passed


def test_hu_window_endpoints_map_exactly_to_output_range() -> None:
    image = sitk.GetImageFromArray(
        np.asarray([[[-1200.0, -1000.0, 400.0, 900.0]]], dtype=np.float32)
    )

    scaled = window_and_scale_ct(
        image,
        hu_window=(-1000.0, 400.0),
        output_range=(-1.0, 1.0),
    )

    values = sitk.GetArrayFromImage(scaled).ravel()
    assert values == pytest.approx([-1.0, -1.0, 1.0, 1.0])
