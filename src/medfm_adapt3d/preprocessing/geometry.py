"""Physical-Coordinate Operations Shared by CT Images and Nodule Masks."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk

from medfm_adapt3d.preprocessing.config import FloatTriplet


class GeometryMismatchError(ValueError):
    """Raised when an image and its annotation do not share one physical grid."""


@dataclass(frozen=True, slots=True)
class SpatialGeometry:
    """Serializable description of a three-dimensional SimpleITK grid."""

    size_xyz: tuple[int, int, int]
    spacing_mm_xyz: FloatTriplet
    origin_mm_lps: FloatTriplet
    direction_lps: tuple[float, ...]

    @classmethod
    def from_image(cls, image: sitk.Image) -> SpatialGeometry:
        validate_scalar_3d_image(image, name="image")
        return cls(
            size_xyz=tuple(int(value) for value in image.GetSize()),
            spacing_mm_xyz=tuple(float(value) for value in image.GetSpacing()),
            origin_mm_lps=tuple(float(value) for value in image.GetOrigin()),
            direction_lps=tuple(float(value) for value in image.GetDirection()),
        )


def validate_scalar_3d_image(image: sitk.Image, *, name: str) -> None:
    """Rejecting non-scalar, non-finite, degenerate, or non-orthonormal grids."""
    if image.GetDimension() != 3:
        raise ValueError(f"{name} must be three-dimensional.")
    if image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError(f"{name} must contain one scalar component per voxel.")

    size = image.GetSize()
    if any(value <= 0 for value in size):
        raise ValueError(f"{name} has a degenerate voxel grid: {size!r}.")

    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    origin = np.asarray(image.GetOrigin(), dtype=np.float64)
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)

    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError(f"{name} spacing must be finite and positive.")
    if not np.all(np.isfinite(origin)):
        raise ValueError(f"{name} origin must be finite.")
    if not np.all(np.isfinite(direction)):
        raise ValueError(f"{name} direction matrix must be finite.")
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} direction matrix must be orthonormal.")
    if not math.isclose(abs(float(np.linalg.det(direction))), 1.0, abs_tol=1e-5):
        raise ValueError(f"{name} direction matrix must describe a rigid orientation.")


def geometry_differences(
    left: sitk.Image,
    right: sitk.Image,
    *,
    tolerance: float = 1e-5,
) -> tuple[str, ...]:
    """Returning explicit grid differences instead of relying on array shape alone."""
    differences: list[str] = []

    if left.GetSize() != right.GetSize():
        differences.append("size")

    for name, left_values, right_values in (
        ("spacing", left.GetSpacing(), right.GetSpacing()),
        ("origin", left.GetOrigin(), right.GetOrigin()),
        ("direction", left.GetDirection(), right.GetDirection()),
    ):
        if not np.allclose(left_values, right_values, rtol=0.0, atol=tolerance):
            differences.append(name)

    return tuple(differences)


def assert_same_geometry(
    image: sitk.Image,
    mask: sitk.Image,
    *,
    tolerance: float = 1e-5,
) -> None:
    """Requiring exact image/mask correspondence in physical patient space."""
    validate_scalar_3d_image(image, name="image")
    validate_scalar_3d_image(mask, name="mask")
    differences = geometry_differences(image, mask, tolerance=tolerance)

    if differences:
        joined = ", ".join(differences)
        raise GeometryMismatchError(
            f"Image/mask physical geometry mismatch in: {joined}."
        )


def orient_to_lps(image: sitk.Image) -> sitk.Image:
    """Permuting and flipping voxel axes into the DICOM LPS orientation convention."""
    validate_scalar_3d_image(image, name="image")
    oriented = sitk.DICOMOrient(image, "LPS")
    validate_scalar_3d_image(oriented, name="LPS-oriented image")
    return oriented


def build_axis_aligned_lps_reference(
    image: sitk.Image,
    target_spacing_mm: FloatTriplet,
) -> sitk.Image:
    """Creating an LPS grid covering the complete physical extent of an image."""
    validate_scalar_3d_image(image, name="image")
    spacing = np.asarray(target_spacing_mm, dtype=np.float64)

    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("target_spacing_mm must contain three finite positive values.")

    size = image.GetSize()
    continuous_edges = [(-0.5, float(length) - 0.5) for length in size]
    physical_corners = np.asarray(
        [
            image.TransformContinuousIndexToPhysicalPoint(index)
            for index in itertools.product(*continuous_edges)
        ],
        dtype=np.float64,
    )

    lower = physical_corners.min(axis=0)
    upper = physical_corners.max(axis=0)
    output_size = np.ceil((upper - lower) / spacing).astype(np.int64)

    if np.any(output_size <= 0):
        raise ValueError("Unable to construct a non-degenerate LPS reference grid.")

    reference = sitk.Image(
        [int(value) for value in output_size],
        sitk.sitkFloat32,
    )
    reference.SetSpacing(tuple(float(value) for value in spacing))
    reference.SetOrigin(tuple(float(value) for value in lower + spacing / 2.0))
    reference.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return reference


def resample_ct(
    image: sitk.Image,
    reference: sitk.Image,
    *,
    outside_hu: float,
) -> sitk.Image:
    """Resampling continuous CT signal with trilinear interpolation."""
    return sitk.Resample(
        image,
        reference,
        sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkLinear,
        float(outside_hu),
        sitk.sitkFloat32,
    )


def resample_mask(mask: sitk.Image, reference: sitk.Image) -> sitk.Image:
    """Resampling discrete labels without manufacturing fractional classes."""
    return sitk.Resample(
        mask,
        reference,
        sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )


def physical_point_is_inside(
    image: sitk.Image,
    point_lps_mm: FloatTriplet,
    *,
    tolerance_voxels: float = 1e-6,
) -> bool:
    """Test containment using the image's continuous-index voxel boundaries."""
    index = image.TransformPhysicalPointToContinuousIndex(point_lps_mm)
    return all(
        -0.5 - tolerance_voxels
        <= coordinate
        <= float(length) - 0.5 + tolerance_voxels
        for coordinate, length in zip(index, image.GetSize(), strict=True)
    )


def crop_around_physical_point(
    image: sitk.Image,
    *,
    center_lps_mm: FloatTriplet,
    size_xyz: tuple[int, int, int],
    outside_value: float,
    output_pixel_id: int,
) -> tuple[sitk.Image, float]:
    """Extracting an odd-sized candidate crop, padding safely at scan boundaries."""
    validate_scalar_3d_image(image, name="image")

    if any(size <= 0 or size % 2 == 0 for size in size_xyz):
        raise ValueError("Candidate crop dimensions must be positive odd integers.")
    if not physical_point_is_inside(image, center_lps_mm):
        raise ValueError("Candidate centre lies outside the source CT field of view.")

    continuous_index = image.TransformPhysicalPointToContinuousIndex(center_lps_mm)
    center_index = tuple(math.floor(value + 0.5) for value in continuous_index)
    radius = tuple(size // 2 for size in size_xyz)
    start = tuple(
        center - half
        for center, half in zip(center_index, radius, strict=True)
    )

    lower_padding = tuple(max(0, -value) for value in start)
    upper_padding = tuple(
        max(0, value + length - available)
        for value, length, available in zip(
            start,
            size_xyz,
            image.GetSize(),
            strict=True,
        )
    )

    padded = sitk.ConstantPad(
        sitk.Cast(image, output_pixel_id),
        lower_padding,
        upper_padding,
        float(outside_value),
    )
    padded_start = tuple(
        value + pad
        for value, pad in zip(start, lower_padding, strict=True)
    )
    crop = sitk.RegionOfInterest(padded, size_xyz, padded_start)

    crop_center_index = tuple(size // 2 for size in size_xyz)
    represented_center = crop.TransformIndexToPhysicalPoint(crop_center_index)
    centering_error_mm = float(
        np.linalg.norm(
            np.asarray(represented_center, dtype=np.float64)
            - np.asarray(center_lps_mm, dtype=np.float64)
        )
    )
    return crop, centering_error_mm
