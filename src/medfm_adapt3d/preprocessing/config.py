"""Validated Configuration for Clinically Faithful CT Preprocessing."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

FloatPair = tuple[float, float]
FloatTriplet = tuple[float, float, float]


def _float_sequence(
    value: object,
    *,
    length: int,
    field: str,
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field} must be a sequence of {length} finite numbers.")

    if len(value) != length:
        raise ValueError(f"{field} must contain exactly {length} values.")

    converted = tuple(float(item) for item in value)

    if not all(math.isfinite(item) for item in converted):
        raise ValueError(f"{field} values must be finite.")

    return converted


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping.")
    return value


def _boolean(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be a YAML boolean.")
    return value


@dataclass(frozen=True, slots=True)
class DICOMValidationPolicy:
    """Metadata and voxel-value checks required before a series is accepted."""

    require_explicit_rescale: bool = True
    maximum_slice_spacing_deviation_mm: float = 0.05
    maximum_in_plane_position_residual_mm: float = 0.10
    minimum_hu_dynamic_range: float = 500.0
    air_reference_upper_bound_hu: float = -500.0

    def __post_init__(self) -> None:
        positive_fields = {
            "maximum_slice_spacing_deviation_mm": self.maximum_slice_spacing_deviation_mm,
            "maximum_in_plane_position_residual_mm": (
                self.maximum_in_plane_position_residual_mm
            ),
            "minimum_hu_dynamic_range": self.minimum_hu_dynamic_range,
        }
        for name, value in positive_fields.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")

        if not math.isfinite(self.air_reference_upper_bound_hu):
            raise ValueError("air_reference_upper_bound_hu must be finite.")


@dataclass(frozen=True, slots=True)
class GeometryPreservationCriteria:
    """Predeclared tolerances for auditing small-mask survival."""

    minimum_resampled_volume_ratio: float = 0.65
    maximum_resampled_volume_ratio: float = 1.50
    tiny_nodule_maximum_source_volume_mm3: float = 150.0
    tiny_nodule_minimum_resampled_volume_ratio: float = 0.45
    tiny_nodule_maximum_resampled_volume_ratio: float = 2.10
    maximum_centroid_shift_mm: float = 1.50
    minimum_crop_volume_retention: float = 0.999

    def __post_init__(self) -> None:
        values = (
            self.minimum_resampled_volume_ratio,
            self.maximum_resampled_volume_ratio,
            self.tiny_nodule_maximum_source_volume_mm3,
            self.tiny_nodule_minimum_resampled_volume_ratio,
            self.tiny_nodule_maximum_resampled_volume_ratio,
            self.maximum_centroid_shift_mm,
            self.minimum_crop_volume_retention,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Geometry-preservation criteria must be finite.")

        if self.minimum_resampled_volume_ratio <= 0:
            raise ValueError("minimum_resampled_volume_ratio must be positive.")
        if self.maximum_resampled_volume_ratio < self.minimum_resampled_volume_ratio:
            raise ValueError(
                "maximum_resampled_volume_ratio must not be smaller than the minimum."
            )
        if self.tiny_nodule_maximum_source_volume_mm3 <= 0:
            raise ValueError("tiny_nodule_maximum_source_volume_mm3 must be positive.")
        if self.tiny_nodule_minimum_resampled_volume_ratio <= 0:
            raise ValueError(
                "tiny_nodule_minimum_resampled_volume_ratio must be positive."
            )
        if (
            self.tiny_nodule_maximum_resampled_volume_ratio
            < self.tiny_nodule_minimum_resampled_volume_ratio
        ):
            raise ValueError(
                "Tiny-nodule maximum volume ratio must not be smaller than its minimum."
            )
        if self.maximum_centroid_shift_mm < 0:
            raise ValueError("maximum_centroid_shift_mm must be non-negative.")
        if not 0 < self.minimum_crop_volume_retention <= 1:
            raise ValueError("minimum_crop_volume_retention must lie in (0, 1].")


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """Deterministic spatial and intensity contract for one CT candidate."""

    target_spacing_mm: FloatTriplet = (1.0, 1.0, 1.0)
    crop_extent_mm: FloatTriplet = (64.0, 64.0, 64.0)
    hu_window: FloatPair = (-1000.0, 400.0)
    output_range: FloatPair = (-1.0, 1.0)
    canonical_orientation: str = "LPS"
    image_interpolation: str = "linear"
    mask_interpolation: str = "nearest"
    fail_on_geometry_qc: bool = True

    def __post_init__(self) -> None:
        for name, values in (
            ("target_spacing_mm", self.target_spacing_mm),
            ("crop_extent_mm", self.crop_extent_mm),
        ):
            if len(values) != 3 or not all(
                math.isfinite(value) and value > 0 for value in values
            ):
                raise ValueError(f"{name} must contain three finite positive values.")

        if len(self.hu_window) != 2 or not all(
            math.isfinite(value) for value in self.hu_window
        ):
            raise ValueError("hu_window must contain two finite values.")
        if self.hu_window[0] >= self.hu_window[1]:
            raise ValueError("hu_window lower bound must be below its upper bound.")

        if len(self.output_range) != 2 or not all(
            math.isfinite(value) for value in self.output_range
        ):
            raise ValueError("output_range must contain two finite values.")
        if self.output_range[0] >= self.output_range[1]:
            raise ValueError("output_range lower bound must be below its upper bound.")

        if self.canonical_orientation != "LPS":
            raise ValueError("The preprocessing coordinate system is locked to DICOM LPS.")
        if self.image_interpolation != "linear":
            raise ValueError("CT intensities must use linear interpolation.")
        if self.mask_interpolation != "nearest":
            raise ValueError("Discrete masks must use nearest-neighbour interpolation.")

    def crop_size_voxels(
        self,
        spacing_mm: FloatTriplet | None = None,
    ) -> tuple[int, int, int]:
        """Returning odd voxel counts covering at least the requested physical extent."""
        spacing = spacing_mm or self.target_spacing_mm
        sizes: list[int] = []

        for extent, step in zip(self.crop_extent_mm, spacing, strict=True):
            intervals = math.ceil(extent / step)
            size = intervals + 1
            if size % 2 == 0:
                size += 1
            sizes.append(size)

        return sizes[0], sizes[1], sizes[2]


@dataclass(frozen=True, slots=True)
class PreprocessingProtocol:
    """Completing validated protocol loaded from one version-controlled file."""

    schema_version: str
    dicom: DICOMValidationPolicy
    preprocessing: PreprocessingConfig
    geometry_qc: GeometryPreservationCriteria


def load_preprocessing_protocol(path: str | Path) -> PreprocessingProtocol:
    """Loading the preprocessing contract without accepting implicit defaults in YAML."""
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = _mapping(payload, field="protocol")
    dicom = _mapping(root.get("dicom_validation"), field="dicom_validation")
    spatial = _mapping(root.get("spatial_preprocessing"), field="spatial_preprocessing")
    intensity = _mapping(root.get("intensity_preprocessing"), field="intensity_preprocessing")
    mask_qc = _mapping(root.get("mask_geometry_qc"), field="mask_geometry_qc")

    schema_version = str(root.get("schema_version", "")).strip()
    if not schema_version:
        raise ValueError("schema_version is required.")

    dicom_policy = DICOMValidationPolicy(
        require_explicit_rescale=_boolean(
            dicom["require_explicit_rescale"],
            field="require_explicit_rescale",
        ),
        maximum_slice_spacing_deviation_mm=float(
            dicom["maximum_slice_spacing_deviation_mm"]
        ),
        maximum_in_plane_position_residual_mm=float(
            dicom["maximum_in_plane_position_residual_mm"]
        ),
        minimum_hu_dynamic_range=float(dicom["minimum_hu_dynamic_range"]),
        air_reference_upper_bound_hu=float(dicom["air_reference_upper_bound_hu"]),
    )

    preprocessing = PreprocessingConfig(
        target_spacing_mm=_float_sequence(
            spatial["target_spacing_mm"],
            length=3,
            field="target_spacing_mm",
        ),
        crop_extent_mm=_float_sequence(
            spatial["candidate_crop_extent_mm"],
            length=3,
            field="candidate_crop_extent_mm",
        ),
        hu_window=_float_sequence(
            intensity["hu_window"],
            length=2,
            field="hu_window",
        ),
        output_range=_float_sequence(
            intensity["output_range"],
            length=2,
            field="output_range",
        ),
        canonical_orientation=str(spatial["canonical_orientation"]),
        image_interpolation=str(spatial["image_interpolation"]),
        mask_interpolation=str(spatial["mask_interpolation"]),
        fail_on_geometry_qc=_boolean(
            mask_qc["fail_on_violation"],
            field="fail_on_violation",
        ),
    )

    criteria = GeometryPreservationCriteria(
        minimum_resampled_volume_ratio=float(
            mask_qc["minimum_resampled_volume_ratio"]
        ),
        maximum_resampled_volume_ratio=float(
            mask_qc["maximum_resampled_volume_ratio"]
        ),
        tiny_nodule_maximum_source_volume_mm3=float(
            mask_qc["tiny_nodule_maximum_source_volume_mm3"]
        ),
        tiny_nodule_minimum_resampled_volume_ratio=float(
            mask_qc["tiny_nodule_minimum_resampled_volume_ratio"]
        ),
        tiny_nodule_maximum_resampled_volume_ratio=float(
            mask_qc["tiny_nodule_maximum_resampled_volume_ratio"]
        ),
        maximum_centroid_shift_mm=float(mask_qc["maximum_centroid_shift_mm"]),
        minimum_crop_volume_retention=float(
            mask_qc["minimum_crop_volume_retention"]
        ),
    )

    return PreprocessingProtocol(
        schema_version=schema_version,
        dicom=dicom_policy,
        preprocessing=preprocessing,
        geometry_qc=criteria,
    )
