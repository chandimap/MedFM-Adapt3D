"""Quantitative Quality Control for Discrete Pulmonary-Nodule Geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk

from medfm_adapt3d.preprocessing.config import GeometryPreservationCriteria
from medfm_adapt3d.preprocessing.geometry import validate_scalar_3d_image


class GeometryPreservationError(ValueError):
    """Raised when resampling or cropping violates predeclared mask tolerances."""


@dataclass(frozen=True, slots=True)
class MaskGeometry:
    """Physical measurements for one non-empty binary nodule mask."""

    voxel_count: int
    volume_mm3: float
    centroid_mm_lps: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class MaskGeometryAudit:
    """Evidence that a mask remains meaningful after resampling and cropping."""

    source: MaskGeometry
    resampled: MaskGeometry
    cropped: MaskGeometry
    resampled_volume_ratio: float
    centroid_shift_mm: float
    crop_volume_retention: float
    tiny_nodule_rule_applied: bool
    applied_minimum_volume_ratio: float
    applied_maximum_volume_ratio: float
    passed: bool
    violations: tuple[str, ...]


def validate_binary_mask(mask: sitk.Image, *, require_non_empty: bool = True) -> None:
    """Rejecting fractional labels, unexpected classes, and empty nodule masks."""
    validate_scalar_3d_image(mask, name="mask")
    invalid = sitk.And(sitk.NotEqual(mask, 0), sitk.NotEqual(mask, 1))
    invalid_stats = sitk.StatisticsImageFilter()
    invalid_stats.Execute(invalid)

    if invalid_stats.GetMaximum() != 0:
        raise ValueError("Mask must contain binary values {0, 1} only.")

    foreground = sitk.StatisticsImageFilter()
    foreground.Execute(sitk.Cast(sitk.Equal(mask, 1), sitk.sitkUInt8))
    if require_non_empty and foreground.GetSum() <= 0:
        raise ValueError("Nodule mask must contain at least one foreground voxel.")


def measure_mask_geometry(mask: sitk.Image) -> MaskGeometry:
    """Measuring mask volume and centroid directly in physical LPS coordinates."""
    validate_binary_mask(mask, require_non_empty=True)
    binary = sitk.Cast(sitk.Equal(mask, 1), sitk.sitkUInt8)
    shape = sitk.LabelShapeStatisticsImageFilter()
    shape.Execute(binary)

    voxel_count = int(shape.GetNumberOfPixels(1))
    voxel_volume = float(np.prod(np.asarray(mask.GetSpacing(), dtype=np.float64)))
    centroid = tuple(float(value) for value in shape.GetCentroid(1))
    return MaskGeometry(
        voxel_count=voxel_count,
        volume_mm3=voxel_count * voxel_volume,
        centroid_mm_lps=centroid,
    )


def audit_mask_geometry(
    source_mask: sitk.Image,
    resampled_mask: sitk.Image,
    cropped_mask: sitk.Image,
    *,
    criteria: GeometryPreservationCriteria,
) -> MaskGeometryAudit:
    """Evaluating small-lesion survival against a protocol fixed before experiments."""
    source = measure_mask_geometry(source_mask)
    resampled = measure_mask_geometry(resampled_mask)
    cropped = measure_mask_geometry(cropped_mask)

    resampled_volume_ratio = resampled.volume_mm3 / source.volume_mm3
    centroid_shift_mm = math.dist(
        source.centroid_mm_lps,
        resampled.centroid_mm_lps,
    )
    crop_volume_retention = cropped.volume_mm3 / resampled.volume_mm3
    tiny_nodule_rule_applied = (
        source.volume_mm3 <= criteria.tiny_nodule_maximum_source_volume_mm3
    )
    minimum_volume_ratio = (
        criteria.tiny_nodule_minimum_resampled_volume_ratio
        if tiny_nodule_rule_applied
        else criteria.minimum_resampled_volume_ratio
    )
    maximum_volume_ratio = (
        criteria.tiny_nodule_maximum_resampled_volume_ratio
        if tiny_nodule_rule_applied
        else criteria.maximum_resampled_volume_ratio
    )

    violations: list[str] = []
    if resampled_volume_ratio < minimum_volume_ratio:
        violations.append("resampled_volume_ratio_below_minimum")
    if resampled_volume_ratio > maximum_volume_ratio:
        violations.append("resampled_volume_ratio_above_maximum")
    if centroid_shift_mm > criteria.maximum_centroid_shift_mm:
        violations.append("centroid_shift_above_maximum")
    if crop_volume_retention < criteria.minimum_crop_volume_retention:
        violations.append("crop_volume_retention_below_minimum")

    return MaskGeometryAudit(
        source=source,
        resampled=resampled,
        cropped=cropped,
        resampled_volume_ratio=resampled_volume_ratio,
        centroid_shift_mm=centroid_shift_mm,
        crop_volume_retention=crop_volume_retention,
        tiny_nodule_rule_applied=tiny_nodule_rule_applied,
        applied_minimum_volume_ratio=minimum_volume_ratio,
        applied_maximum_volume_ratio=maximum_volume_ratio,
        passed=not violations,
        violations=tuple(violations),
    )


def enforce_geometry_audit(audit: MaskGeometryAudit) -> None:
    """Stopping an experiment before a failed geometric transformation is consumed."""
    if audit.passed:
        return

    joined = ", ".join(audit.violations)
    raise GeometryPreservationError(
        "Pulmonary-nodule geometry failed preprocessing QC: " + joined
    )
