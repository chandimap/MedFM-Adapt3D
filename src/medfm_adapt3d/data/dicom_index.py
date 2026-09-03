"""Header-only DICOM indexing for auditable LIDC-IDRI CT cohort construction."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import pydicom

from medfm_adapt3d.data.lidc_schema import DICOMSeriesSummary

_REQUIRED_TAGS = [
    "PatientID",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "Modality",
    "Rows",
    "Columns",
    "PixelSpacing",
    "SliceThickness",
    "ImagePositionPatient",
    "Manufacturer",
    "ManufacturerModelName",
    "ConvolutionKernel",
    "KVP",
]


def _clean_optional_text(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    return text or None


def _required_text(dataset: pydicom.Dataset, field: str, path: Path) -> str:
    value = _clean_optional_text(getattr(dataset, field, None))

    if value is None:
        raise ValueError(f"{path}: missing required DICOM attribute {field}.")

    return value


def _required_positive_float(
    dataset: pydicom.Dataset,
    field: str,
    path: Path,
) -> float:
    value = getattr(dataset, field, None)

    if value is None:
        raise ValueError(f"{path}: missing required DICOM attribute {field}.")

    numeric = float(value)

    if numeric <= 0:
        raise ValueError(f"{path}: {field} must be positive, got {numeric}.")

    return numeric


def _read_header(path: Path) -> pydicom.Dataset:
    return pydicom.dcmread(
        path,
        stop_before_pixels=True,
        specific_tags=_REQUIRED_TAGS,
        force=False,
    )


def _candidate_dicom_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def index_dicom_series(
    root: str | Path,
) -> dict[tuple[str, str], DICOMSeriesSummary]:
    """Index DICOM files by exact (StudyInstanceUID, SeriesInstanceUID)."""
    dicom_root = Path(root)

    if not dicom_root.exists():
        raise FileNotFoundError(dicom_root)

    grouped: dict[tuple[str, str], list[tuple[Path, pydicom.Dataset]]] = defaultdict(list)

    for path in _candidate_dicom_files(dicom_root):
        try:
            dataset = _read_header(path)
        except (pydicom.errors.InvalidDicomError, OSError):
            continue

        modality = _clean_optional_text(getattr(dataset, "Modality", None))
        if modality != "CT":
            continue

        study_uid = _required_text(dataset, "StudyInstanceUID", path)
        series_uid = _required_text(dataset, "SeriesInstanceUID", path)

        grouped[(study_uid, series_uid)].append((path, dataset))

    summaries: dict[tuple[str, str], DICOMSeriesSummary] = {}

    for key, members in grouped.items():
        summaries[key] = _summarise_series(members)

    return summaries


def _summarise_series(
    members: list[tuple[Path, pydicom.Dataset]],
) -> DICOMSeriesSummary:
    paths = [path for path, _ in members]
    datasets = [dataset for _, dataset in members]

    patient_ids = {
        _required_text(dataset, "PatientID", path)
        for path, dataset in members
    }
    study_uids = {
        _required_text(dataset, "StudyInstanceUID", path)
        for path, dataset in members
    }
    series_uids = {
        _required_text(dataset, "SeriesInstanceUID", path)
        for path, dataset in members
    }

    if len(patient_ids) != 1:
        raise ValueError(f"Series contains multiple PatientID values: {patient_ids}")

    if len(study_uids) != 1:
        raise ValueError(f"Series contains multiple StudyInstanceUID values: {study_uids}")

    if len(series_uids) != 1:
        raise ValueError(
            f"Series contains multiple SeriesInstanceUID values: {series_uids}"
        )

    sop_uids = tuple(
        _required_text(dataset, "SOPInstanceUID", path)
        for path, dataset in members
    )

    if len(set(sop_uids)) != len(sop_uids):
        raise ValueError("Duplicate SOPInstanceUID detected within CT series.")

    rows = {int(dataset.Rows) for dataset in datasets}
    columns = {int(dataset.Columns) for dataset in datasets}

    if len(rows) != 1 or len(columns) != 1:
        raise ValueError("Rows/Columns are inconsistent within CT series.")

    pixel_spacings = {
        (
            float(dataset.PixelSpacing[0]),
            float(dataset.PixelSpacing[1]),
        )
        for dataset in datasets
    }

    if len(pixel_spacings) != 1:
        raise ValueError("PixelSpacing is inconsistent within CT series.")

    pixel_spacing = next(iter(pixel_spacings))

    if pixel_spacing[0] <= 0 or pixel_spacing[1] <= 0:
        raise ValueError("PixelSpacing values must be positive.")

    thicknesses = {
        _required_positive_float(dataset, "SliceThickness", path)
        for path, dataset in members
    }

    if len(thicknesses) != 1:
        raise ValueError("SliceThickness is inconsistent within CT series.")

    z_positions = []

    for path, dataset in members:
        position = getattr(dataset, "ImagePositionPatient", None)

        if position is None or len(position) != 3:
            raise ValueError(f"{path}: invalid ImagePositionPatient.")

        z_positions.append(float(position[2]))

    ordered_z = tuple(sorted(z_positions))

    if len(set(ordered_z)) < 2:
        raise ValueError("CT series must contain at least two distinct z positions.")

    first = datasets[0]

    source_directory = str(Path(paths[0]).parent)

    kvp_value = getattr(first, "KVP", None)

    return DICOMSeriesSummary(
        patient_id=next(iter(patient_ids)),
        study_instance_uid=next(iter(study_uids)),
        series_instance_uid=next(iter(series_uids)),
        modality="CT",
        sop_instance_uids=tuple(sorted(sop_uids)),
        rows=next(iter(rows)),
        columns=next(iter(columns)),
        pixel_spacing_mm=pixel_spacing,
        slice_thickness_mm=next(iter(thicknesses)),
        z_positions_mm=ordered_z,
        manufacturer=_clean_optional_text(getattr(first, "Manufacturer", None)),
        manufacturer_model_name=_clean_optional_text(
            getattr(first, "ManufacturerModelName", None)
        ),
        convolution_kernel=_clean_optional_text(
            getattr(first, "ConvolutionKernel", None)
        ),
        kvp=float(kvp_value) if kvp_value is not None else None,
        source_directory=source_directory,
    )