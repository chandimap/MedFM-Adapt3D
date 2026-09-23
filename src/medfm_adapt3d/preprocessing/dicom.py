"""Auditable Conversion of Single-Frame CT DICOM Series into HU Volumes."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
import SimpleITK as sitk
from pydicom.dataset import Dataset

from medfm_adapt3d.preprocessing.config import DICOMValidationPolicy, FloatTriplet
from medfm_adapt3d.preprocessing.geometry import validate_scalar_3d_image


class DICOMValidationError(ValueError):
    """Raised when a DICOM series cannot support trustworthy HU geometry."""


@dataclass(frozen=True, slots=True)
class DICOMSeriesAudit:
    """Non-pixel provenance proving how one CT series was reconstructed."""

    study_instance_uid: str
    series_instance_uid: str
    number_of_slices: int
    ordered_sop_instance_uids: tuple[str, ...]
    spacing_mm_xyz: FloatTriplet
    direction_lps: tuple[float, ...]
    rescale_slopes: tuple[float, ...]
    rescale_intercepts: tuple[float, ...]
    minimum_hu: float
    maximum_hu: float
    hu_conversion: str = "stored_value * RescaleSlope + RescaleIntercept"


@dataclass(frozen=True, slots=True)
class _HeaderRecord:
    path: Path


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_text(dataset: Dataset, field: str, path: Path) -> str:
    value = _clean_text(getattr(dataset, field, None))
    if value is None:
        raise DICOMValidationError(f"{path}: missing required DICOM field {field}.")
    return value


def _required_float_sequence(
    dataset: Dataset,
    field: str,
    *,
    length: int,
    path: Path,
) -> np.ndarray:
    value = getattr(dataset, field, None)
    if value is None:
        raise DICOMValidationError(
            f"{path}: {field} must contain exactly {length} values."
        )

    try:
        items = tuple(value)
    except TypeError as error:
        raise DICOMValidationError(
            f"{path}: {field} must be a DICOM multi-value field."
        ) from error

    if len(items) != length:
        raise DICOMValidationError(
            f"{path}: {field} must contain exactly {length} values."
        )

    try:
        converted = np.asarray([float(item) for item in items], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise DICOMValidationError(
            f"{path}: {field} contains non-numeric values."
        ) from error

    if not np.all(np.isfinite(converted)):
        raise DICOMValidationError(f"{path}: {field} contains non-finite values.")
    return converted


def rescale_stored_pixels_to_hu(
    stored_pixels: np.ndarray,
    *,
    slope: float,
    intercept: float,
) -> np.ndarray:
    """Applying the DICOM modality rescale explicitly and return float32 HU."""
    numeric_slope = float(slope)
    numeric_intercept = float(intercept)

    if not math.isfinite(numeric_slope) or numeric_slope == 0:
        raise DICOMValidationError("RescaleSlope must be finite and non-zero.")
    if not math.isfinite(numeric_intercept):
        raise DICOMValidationError("RescaleIntercept must be finite.")

    pixels = np.asarray(stored_pixels)
    if pixels.ndim != 2:
        raise DICOMValidationError("Each source DICOM instance must contain one 2D frame.")
    if not np.all(np.isfinite(pixels)):
        raise DICOMValidationError("Stored CT pixels contain non-finite values.")

    return (
        pixels.astype(np.float32) * np.float32(numeric_slope)
        + np.float32(numeric_intercept)
    )


def _discover_ct_headers(root: Path) -> dict[str, list[_HeaderRecord]]:
    grouped: dict[str, list[_HeaderRecord]] = defaultdict(list)

    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        try:
            dataset = pydicom.dcmread(path, stop_before_pixels=True, force=False)
        except (pydicom.errors.InvalidDicomError, OSError):
            continue

        if _clean_text(getattr(dataset, "Modality", None)) != "CT":
            continue

        series_uid = _required_text(dataset, "SeriesInstanceUID", path)
        grouped[series_uid].append(_HeaderRecord(path=path))

    return dict(grouped)


def _select_series(
    grouped: dict[str, list[_HeaderRecord]],
    requested_uid: str | None,
) -> tuple[str, list[_HeaderRecord]]:
    if not grouped:
        raise DICOMValidationError("No readable CT DICOM series was found.")

    if requested_uid is not None:
        if requested_uid not in grouped:
            raise DICOMValidationError(
                f"Requested SeriesInstanceUID {requested_uid!r} was not found."
            )
        return requested_uid, grouped[requested_uid]

    if len(grouped) != 1:
        available = ", ".join(sorted(grouped))
        raise DICOMValidationError(
            "The directory contains multiple CT series; provide series_instance_uid. "
            f"Available values: {available}."
        )

    selected_uid = next(iter(grouped))
    return selected_uid, grouped[selected_uid]


def _rescale_parameters(
    dataset: Dataset,
    *,
    path: Path,
    require_explicit: bool,
) -> tuple[float, float]:
    slope_value = getattr(dataset, "RescaleSlope", None)
    intercept_value = getattr(dataset, "RescaleIntercept", None)

    if require_explicit and (slope_value is None or intercept_value is None):
        raise DICOMValidationError(
            f"{path}: explicit RescaleSlope and RescaleIntercept are required."
        )

    slope = float(1.0 if slope_value is None else slope_value)
    intercept = float(0.0 if intercept_value is None else intercept_value)

    rescale_type = _clean_text(getattr(dataset, "RescaleType", None))
    if rescale_type is not None and rescale_type.upper() != "HU":
        raise DICOMValidationError(
            f"{path}: unsupported CT RescaleType {rescale_type!r}; expected HU."
        )

    if not math.isfinite(slope) or slope == 0:
        raise DICOMValidationError(f"{path}: invalid RescaleSlope {slope!r}.")
    if not math.isfinite(intercept):
        raise DICOMValidationError(f"{path}: invalid RescaleIntercept {intercept!r}.")

    return slope, intercept


def load_ct_dicom_series(
    root: str | Path,
    *,
    series_instance_uid: str | None = None,
    policy: DICOMValidationPolicy | None = None,
) -> tuple[sitk.Image, DICOMSeriesAudit]:
    """Loading one regular single-frame CT series with explicit HU conversion."""
    source = Path(root)
    if not source.is_dir():
        raise FileNotFoundError(source)

    active_policy = policy or DICOMValidationPolicy()
    selected_uid, headers = _select_series(
        _discover_ct_headers(source),
        series_instance_uid,
    )

    datasets: list[tuple[Path, Dataset]] = []
    for header in headers:
        try:
            dataset = pydicom.dcmread(header.path, force=False)
            _ = dataset.pixel_array
        except Exception as error:
            raise DICOMValidationError(
                f"{header.path}: unable to decode CT PixelData."
            ) from error
        datasets.append((header.path, dataset))

    patient_ids = {
        _required_text(dataset, "PatientID", path) for path, dataset in datasets
    }
    study_uids = {
        _required_text(dataset, "StudyInstanceUID", path) for path, dataset in datasets
    }
    series_uids = {
        _required_text(dataset, "SeriesInstanceUID", path) for path, dataset in datasets
    }
    sop_uids = [
        _required_text(dataset, "SOPInstanceUID", path) for path, dataset in datasets
    ]

    if len(patient_ids) != 1:
        raise DICOMValidationError("One CT series contains multiple PatientID values.")
    if len(study_uids) != 1:
        raise DICOMValidationError("One CT series contains multiple StudyInstanceUID values.")
    if series_uids != {selected_uid}:
        raise DICOMValidationError("SeriesInstanceUID changed between header and pixel reads.")
    if len(set(sop_uids)) != len(sop_uids):
        raise DICOMValidationError("Duplicate SOPInstanceUID detected within CT series.")
    if len(datasets) < 2:
        raise DICOMValidationError("A 3D CT series requires at least two DICOM slices.")

    first_path, first = datasets[0]
    rows = int(getattr(first, "Rows", 0))
    columns = int(getattr(first, "Columns", 0))
    if rows <= 0 or columns <= 0:
        raise DICOMValidationError(f"{first_path}: invalid Rows or Columns.")

    first_spacing = _required_float_sequence(
        first,
        "PixelSpacing",
        length=2,
        path=first_path,
    )
    if np.any(first_spacing <= 0):
        raise DICOMValidationError("PixelSpacing values must be positive.")

    first_orientation = _required_float_sequence(
        first,
        "ImageOrientationPatient",
        length=6,
        path=first_path,
    )
    x_direction = first_orientation[:3]
    y_direction = first_orientation[3:]

    if not math.isclose(float(np.linalg.norm(x_direction)), 1.0, abs_tol=1e-4):
        raise DICOMValidationError("First DICOM orientation vector is not unit length.")
    if not math.isclose(float(np.linalg.norm(y_direction)), 1.0, abs_tol=1e-4):
        raise DICOMValidationError("Second DICOM orientation vector is not unit length.")
    if not math.isclose(float(np.dot(x_direction, y_direction)), 0.0, abs_tol=1e-4):
        raise DICOMValidationError("DICOM in-plane orientation vectors are not orthogonal.")

    x_direction = x_direction / np.linalg.norm(x_direction)
    y_direction = y_direction / np.linalg.norm(y_direction)
    slice_direction = np.cross(x_direction, y_direction)
    slice_direction = slice_direction / np.linalg.norm(slice_direction)

    positions: list[np.ndarray] = []
    rescale: list[tuple[float, float]] = []

    for path, dataset in datasets:
        if _clean_text(getattr(dataset, "Modality", None)) != "CT":
            raise DICOMValidationError(f"{path}: non-CT instance in selected series.")
        if int(getattr(dataset, "NumberOfFrames", 1)) != 1:
            raise DICOMValidationError(f"{path}: multi-frame CT is not supported.")
        if int(getattr(dataset, "SamplesPerPixel", 1)) != 1:
            raise DICOMValidationError(f"{path}: CT must contain one sample per pixel.")
        if _clean_text(getattr(dataset, "PhotometricInterpretation", None)) != "MONOCHROME2":
            raise DICOMValidationError(
                f"{path}: only MONOCHROME2 CT images are accepted."
            )
        if int(dataset.Rows) != rows or int(dataset.Columns) != columns:
            raise DICOMValidationError("Rows/Columns are inconsistent within CT series.")

        spacing = _required_float_sequence(
            dataset,
            "PixelSpacing",
            length=2,
            path=path,
        )
        orientation = _required_float_sequence(
            dataset,
            "ImageOrientationPatient",
            length=6,
            path=path,
        )
        if not np.allclose(spacing, first_spacing, rtol=0.0, atol=1e-5):
            raise DICOMValidationError("PixelSpacing is inconsistent within CT series.")
        if not np.allclose(orientation, first_orientation, rtol=0.0, atol=1e-5):
            raise DICOMValidationError(
                "ImageOrientationPatient is inconsistent within CT series."
            )

        positions.append(
            _required_float_sequence(
                dataset,
                "ImagePositionPatient",
                length=3,
                path=path,
            )
        )
        rescale.append(
            _rescale_parameters(
                dataset,
                path=path,
                require_explicit=active_policy.require_explicit_rescale,
            )
        )

    projections = np.asarray(
        [float(np.dot(position, slice_direction)) for position in positions],
        dtype=np.float64,
    )
    order = np.argsort(projections)
    ordered_projections = projections[order]
    differences = np.diff(ordered_projections)

    if np.any(differences <= 0):
        raise DICOMValidationError("Duplicate or reversed slice positions detected.")

    slice_spacing = float(np.median(differences))
    maximum_spacing_error = float(np.max(np.abs(differences - slice_spacing)))
    if maximum_spacing_error > active_policy.maximum_slice_spacing_deviation_mm:
        raise DICOMValidationError(
            "Irregular slice spacing exceeds the declared tolerance: "
            f"{maximum_spacing_error:.6f} mm."
        )

    ordered_positions = [positions[index] for index in order]
    origin = ordered_positions[0]
    for position in ordered_positions:
        displacement = position - origin
        axial_displacement = np.dot(displacement, slice_direction) * slice_direction
        residual = float(np.linalg.norm(displacement - axial_displacement))
        if residual > active_policy.maximum_in_plane_position_residual_mm:
            raise DICOMValidationError(
                "Slice positions contain unsupported in-plane drift or gantry tilt: "
                f"{residual:.6f} mm."
            )

    hu_slices: list[np.ndarray] = []
    for index in order:
        _, dataset = datasets[int(index)]
        slope, intercept = rescale[int(index)]
        hu_slices.append(
            rescale_stored_pixels_to_hu(
                dataset.pixel_array,
                slope=slope,
                intercept=intercept,
            )
        )

    volume = np.stack(hu_slices, axis=0).astype(np.float32, copy=False)
    minimum_hu = float(volume.min())
    maximum_hu = float(volume.max())

    if maximum_hu - minimum_hu < active_policy.minimum_hu_dynamic_range:
        raise DICOMValidationError(
            "CT HU dynamic range is too small for a thoracic acquisition: "
            f"{maximum_hu - minimum_hu:.3f} HU."
        )
    if minimum_hu > active_policy.air_reference_upper_bound_hu:
        raise DICOMValidationError(
            "No expected thoracic air reference was observed after HU conversion: "
            f"minimum {minimum_hu:.3f} HU."
        )

    image = sitk.GetImageFromArray(volume, isVector=False)
    image.SetSpacing(
        (
            float(first_spacing[1]),
            float(first_spacing[0]),
            slice_spacing,
        )
    )
    image.SetOrigin(tuple(float(value) for value in origin))
    direction = np.column_stack((x_direction, y_direction, slice_direction))
    image.SetDirection(tuple(float(value) for value in direction.ravel()))
    validate_scalar_3d_image(image, name="reconstructed CT")

    audit = DICOMSeriesAudit(
        study_instance_uid=next(iter(study_uids)),
        series_instance_uid=selected_uid,
        number_of_slices=len(datasets),
        ordered_sop_instance_uids=tuple(
            sop_uids[int(index)] for index in order
        ),
        spacing_mm_xyz=tuple(float(value) for value in image.GetSpacing()),
        direction_lps=tuple(float(value) for value in image.GetDirection()),
        rescale_slopes=tuple(sorted({float(value[0]) for value in rescale})),
        rescale_intercepts=tuple(sorted({float(value[1]) for value in rescale})),
        minimum_hu=minimum_hu,
        maximum_hu=maximum_hu,
    )
    return image, audit
