"""Typed LIDC-IDRI structures used by the cohort-audit pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class NoduleCharacteristics:
    """Reader-specific semantic assessment for one >=3 mm nodule annotation."""

    subtlety: int | None = None
    internal_structure: int | None = None
    calcification: int | None = None
    sphericity: int | None = None
    margin: int | None = None
    lobulation: int | None = None
    spiculation: int | None = None
    texture: int | None = None
    malignancy: int | None = None


@dataclass(frozen=True, slots=True)
class ROIReference:
    """Reference from an LIDC reader annotation to one DICOM image."""

    sop_instance_uid: str
    z_position_mm: float
    inclusion: bool
    edge_points: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class ReaderNoduleAnnotation:
    """One scan-local reader's annotation of one pulmonary finding."""

    reader_slot: int
    reader_annotation_id: str
    characteristics: NoduleCharacteristics | None
    rois: tuple[ROIReference, ...]

    @property
    def is_large_nodule(self) -> bool:
        """Returning whether the XML annotation carries >=3 mm nodule characteristics."""
        return self.characteristics is not None


@dataclass(frozen=True, slots=True)
class DICOMSeriesSummary:
    """Auditable DICOM metadata for one CT series."""

    patient_id: str
    study_instance_uid: str
    series_instance_uid: str
    modality: str

    sop_instance_uids: tuple[str, ...]

    rows: int
    columns: int

    pixel_spacing_mm: tuple[float, float]
    slice_thickness_mm: float
    z_positions_mm: tuple[float, ...]

    manufacturer: str | None
    manufacturer_model_name: str | None
    convolution_kernel: str | None
    kvp: float | None

    source_directory: str

    @property
    def number_of_slices(self) -> int:
        return len(self.sop_instance_uids)

    @property
    def median_slice_spacing_mm(self) -> float | None:
        if len(self.z_positions_mm) < 2:
            return None

        differences = [
            right - left
            for left, right in zip(
                self.z_positions_mm[:-1],
                self.z_positions_mm[1:],
                strict=True,
            )
        ]

        ordered = sorted(abs(value) for value in differences)
        midpoint = len(ordered) // 2

        if len(ordered) % 2:
            return ordered[midpoint]

        return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


@dataclass(frozen=True, slots=True)
class ManifestQC:
    """Quality-control outcome for one LIDC CT series."""

    passed: bool
    issues: tuple[str, ...]
    annotation_count: int
    large_nodule_annotation_count: int
    annotation_sop_reference_count: int
    missing_annotation_sop_reference_count: int
    meets_luna16_slice_thickness_rule: bool


@dataclass(frozen=True, slots=True)
class LIDCManifestRecord:
    """One auditable LIDC scan record."""

    schema_version: str

    patient_id: str
    study_instance_uid: str
    series_instance_uid: str

    dicom: DICOMSeriesSummary
    annotations: tuple[ReaderNoduleAnnotation, ...]
    qc: ManifestQC

    def to_dict(self) -> dict[str, Any]:
        """Converting a frozen typed record into a JSON-serialisable mapping."""
        return asdict(self)