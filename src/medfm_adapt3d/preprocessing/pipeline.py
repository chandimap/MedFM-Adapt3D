"""Deterministic, Geometry-Audited Preprocessing for 3D CT Candidates."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import SimpleITK as sitk
import torch

from medfm_adapt3d.preprocessing.config import (
    FloatTriplet,
    GeometryPreservationCriteria,
    PreprocessingConfig,
)
from medfm_adapt3d.preprocessing.geometry import (
    SpatialGeometry,
    assert_same_geometry,
    build_axis_aligned_lps_reference,
    crop_around_physical_point,
    orient_to_lps,
    physical_point_is_inside,
    resample_ct,
    resample_mask,
    validate_scalar_3d_image,
)
from medfm_adapt3d.preprocessing.quality import (
    MaskGeometryAudit,
    audit_mask_geometry,
    enforce_geometry_audit,
    validate_binary_mask,
)


@dataclass(frozen=True, slots=True)
class PreprocessingTrace:
    """Machine-readable evidence for one deterministic candidate transform."""

    source_geometry: SpatialGeometry
    output_geometry: SpatialGeometry
    source_hu_range: tuple[float, float]
    output_intensity_range: tuple[float, float]
    candidate_center_mm_lps: FloatTriplet
    crop_centering_error_mm: float
    image_interpolation: str
    mask_interpolation: str | None
    mask_geometry: MaskGeometryAudit | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PreprocessedCandidate:
    """Channel-first model input with its aligned mask and spatial evidence."""

    image: torch.Tensor
    mask: torch.Tensor | None
    trace: PreprocessingTrace


def _intensity_range(image: sitk.Image, *, name: str) -> tuple[float, float]:
    validate_scalar_3d_image(image, name=name)
    array = sitk.GetArrayViewFromImage(image)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return float(array.min()), float(array.max())


def window_and_scale_ct(
    image: sitk.Image,
    *,
    hu_window: tuple[float, float],
    output_range: tuple[float, float],
) -> sitk.Image:
    """Clipping HU values and mapping them linearly onto a declared model-input range."""
    lower_hu, upper_hu = hu_window
    lower_output, upper_output = output_range
    clipped = sitk.Clamp(
        sitk.Cast(image, sitk.sitkFloat32),
        lowerBound=float(lower_hu),
        upperBound=float(upper_hu),
    )
    scaled = (clipped - lower_hu) / (upper_hu - lower_hu)
    scaled = scaled * (upper_output - lower_output) + lower_output
    return sitk.Cast(scaled, sitk.sitkFloat32)


def _to_channel_first_tensor(image: sitk.Image) -> torch.Tensor:
    array_zyx = sitk.GetArrayFromImage(image)
    return torch.from_numpy(array_zyx).unsqueeze(0).contiguous()


class CandidatePreprocessor:
    """Applying one fixed CT/mask transformation contract to a physical candidate."""

    def __init__(
        self,
        config: PreprocessingConfig,
        geometry_criteria: GeometryPreservationCriteria,
    ) -> None:
        self._config = config
        self._geometry_criteria = geometry_criteria

    @property
    def config(self) -> PreprocessingConfig:
        return self._config

    def __call__(
        self,
        image_hu: sitk.Image,
        *,
        candidate_center_mm_lps: FloatTriplet,
        mask: sitk.Image | None = None,
    ) -> PreprocessedCandidate:
        validate_scalar_3d_image(image_hu, name="CT image")
        source_hu_range = _intensity_range(image_hu, name="CT image")
        center = tuple(float(value) for value in candidate_center_mm_lps)

        if len(center) != 3 or not all(math.isfinite(value) for value in center):
            raise ValueError("candidate_center_mm_lps must contain three finite values.")
        if not physical_point_is_inside(image_hu, center):
            raise ValueError("Candidate centre lies outside the source CT field of view.")

        if mask is not None:
            validate_binary_mask(mask, require_non_empty=True)
            assert_same_geometry(image_hu, mask)

        oriented_image = orient_to_lps(sitk.Cast(image_hu, sitk.sitkFloat32))
        oriented_mask = orient_to_lps(mask) if mask is not None else None

        if oriented_mask is not None:
            assert_same_geometry(oriented_image, oriented_mask)

        reference = build_axis_aligned_lps_reference(
            oriented_image,
            self._config.target_spacing_mm,
        )
        resampled_image = resample_ct(
            oriented_image,
            reference,
            outside_hu=self._config.hu_window[0],
        )
        resampled_mask = (
            resample_mask(oriented_mask, reference)
            if oriented_mask is not None
            else None
        )

        if resampled_mask is not None:
            validate_binary_mask(resampled_mask, require_non_empty=True)
            assert_same_geometry(resampled_image, resampled_mask)

        scaled_image = window_and_scale_ct(
            resampled_image,
            hu_window=self._config.hu_window,
            output_range=self._config.output_range,
        )
        crop_size = self._config.crop_size_voxels(
            tuple(float(value) for value in scaled_image.GetSpacing())
        )
        image_crop, centering_error = crop_around_physical_point(
            scaled_image,
            center_lps_mm=center,
            size_xyz=crop_size,
            outside_value=self._config.output_range[0],
            output_pixel_id=sitk.sitkFloat32,
        )

        mask_crop: sitk.Image | None = None
        mask_audit: MaskGeometryAudit | None = None
        if resampled_mask is not None and oriented_mask is not None:
            mask_crop, mask_centering_error = crop_around_physical_point(
                resampled_mask,
                center_lps_mm=center,
                size_xyz=crop_size,
                outside_value=0,
                output_pixel_id=sitk.sitkUInt8,
            )
            if not math.isclose(centering_error, mask_centering_error, abs_tol=1e-8):
                raise RuntimeError("Image and mask crop centres diverged unexpectedly.")

            assert_same_geometry(image_crop, mask_crop)
            mask_audit = audit_mask_geometry(
                mask,
                resampled_mask,
                mask_crop,
                criteria=self._geometry_criteria,
            )
            if self._config.fail_on_geometry_qc:
                enforce_geometry_audit(mask_audit)

        output_range = _intensity_range(image_crop, name="preprocessed CT crop")
        tolerance = 1e-6
        if output_range[0] < self._config.output_range[0] - tolerance:
            raise RuntimeError("Preprocessed intensities fell below the declared range.")
        if output_range[1] > self._config.output_range[1] + tolerance:
            raise RuntimeError("Preprocessed intensities exceeded the declared range.")

        image_tensor = _to_channel_first_tensor(image_crop).to(dtype=torch.float32)
        mask_tensor = (
            _to_channel_first_tensor(mask_crop).to(dtype=torch.uint8)
            if mask_crop is not None
            else None
        )

        trace = PreprocessingTrace(
            source_geometry=SpatialGeometry.from_image(image_hu),
            output_geometry=SpatialGeometry.from_image(image_crop),
            source_hu_range=source_hu_range,
            output_intensity_range=output_range,
            candidate_center_mm_lps=center,
            crop_centering_error_mm=centering_error,
            image_interpolation=self._config.image_interpolation,
            mask_interpolation=(
                self._config.mask_interpolation if mask is not None else None
            ),
            mask_geometry=mask_audit,
        )
        return PreprocessedCandidate(
            image=image_tensor,
            mask=mask_tensor,
            trace=trace,
        )
