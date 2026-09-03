"""Construction and validation of the auditable LIDC-IDRI scan manifest."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from medfm_adapt3d.data.dicom_index import index_dicom_series
from medfm_adapt3d.data.lidc_schema import (
    DICOMSeriesSummary,
    LIDCManifestRecord,
    ManifestQC,
    ReaderNoduleAnnotation,
)
from medfm_adapt3d.data.lidc_xml import LIDCXMLScan, parse_lidc_xml

SCHEMA_VERSION = "1.0"
LUNA16_MAXIMUM_SLICE_THICKNESS_MM = 2.5


def _audit_annotation_links(
    dicom: DICOMSeriesSummary,
    annotations: tuple[ReaderNoduleAnnotation, ...],
) -> ManifestQC:
    sop_uids = set(dicom.sop_instance_uids)

    annotation_references = [
        roi.sop_instance_uid
        for annotation in annotations
        for roi in annotation.rois
    ]

    missing_references = [
        sop_uid
        for sop_uid in annotation_references
        if sop_uid not in sop_uids
    ]

    issues: list[str] = []

    if missing_references:
        issues.append(
            "annotation_sop_reference_missing_from_dicom_series"
        )

    duplicate_z_positions = (
        len(set(dicom.z_positions_mm)) != len(dicom.z_positions_mm)
    )

    if duplicate_z_positions:
        issues.append("duplicate_slice_z_position")

    large_nodule_count = sum(
        annotation.is_large_nodule
        for annotation in annotations
    )

    return ManifestQC(
        passed=not issues,
        issues=tuple(sorted(set(issues))),
        annotation_count=len(annotations),
        large_nodule_annotation_count=large_nodule_count,
        annotation_sop_reference_count=len(annotation_references),
        missing_annotation_sop_reference_count=len(missing_references),
        meets_luna16_slice_thickness_rule=(
            dicom.slice_thickness_mm
            <= LUNA16_MAXIMUM_SLICE_THICKNESS_MM
        ),
    )


def build_manifest_record(
    dicom: DICOMSeriesSummary,
    xml_scan: LIDCXMLScan,
) -> LIDCManifestRecord:
    """Creating one scan-level record after exact UID reconciliation."""
    if dicom.study_instance_uid != xml_scan.study_instance_uid:
        raise ValueError(
            "DICOM/XML StudyInstanceUID mismatch: "
            f"{dicom.study_instance_uid!r} != {xml_scan.study_instance_uid!r}"
        )

    if dicom.series_instance_uid != xml_scan.series_instance_uid:
        raise ValueError(
            "DICOM/XML SeriesInstanceUID mismatch: "
            f"{dicom.series_instance_uid!r} != {xml_scan.series_instance_uid!r}"
        )

    qc = _audit_annotation_links(
        dicom,
        xml_scan.annotations,
    )

    return LIDCManifestRecord(
        schema_version=SCHEMA_VERSION,
        patient_id=dicom.patient_id,
        study_instance_uid=dicom.study_instance_uid,
        series_instance_uid=dicom.series_instance_uid,
        dicom=dicom,
        annotations=xml_scan.annotations,
        qc=qc,
    )


def discover_xml_files(root: str | Path) -> tuple[Path, ...]:
    """Return XML files using deterministic lexical ordering."""
    xml_root = Path(root)

    if not xml_root.exists():
        raise FileNotFoundError(xml_root)

    return tuple(sorted(xml_root.rglob("*.xml")))


def build_lidc_manifest(
    *,
    dicom_root: str | Path,
    xml_root: str | Path,
) -> tuple[LIDCManifestRecord, ...]:
    """Reconciling local TCIA DICOM headers with final LIDC XML annotations."""
    dicom_series = index_dicom_series(dicom_root)

    records: list[LIDCManifestRecord] = []
    observed_xml_keys: set[tuple[str, str]] = set()

    for xml_path in discover_xml_files(xml_root):
        xml_scan = parse_lidc_xml(xml_path)

        key = (
            xml_scan.study_instance_uid,
            xml_scan.series_instance_uid,
        )

        if key in observed_xml_keys:
            raise ValueError(
                "Multiple XML files resolve to the same DICOM study/series: "
                f"{key!r}"
            )

        observed_xml_keys.add(key)

        dicom = dicom_series.get(key)

        if dicom is None:
            # The missing linkage is represented at cohort-build level rather
            # than silently selecting an unrelated CT series.
            continue

        records.append(
            build_manifest_record(
                dicom,
                xml_scan,
            )
        )

    records.sort(
        key=lambda record: (
            record.patient_id,
            record.study_instance_uid,
            record.series_instance_uid,
        )
    )

    return tuple(records)


def write_manifest(
    records: tuple[LIDCManifestRecord, ...],
    destination: str | Path,
) -> None:
    """Writing a deterministic JSON manifest."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = [record.to_dict() for record in records]

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def build_audit_summary(
    records: tuple[LIDCManifestRecord, ...],
) -> dict[str, object]:
    """Producing cohort-level statistics without claiming model performance."""
    patients = {record.patient_id for record in records}

    manufacturers = Counter(
        record.dicom.manufacturer or "UNKNOWN"
        for record in records
    )

    failures = [
        record
        for record in records
        if not record.qc.passed
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "patients": len(patients),
        "ct_series": len(records),
        "qc_passed_series": len(records) - len(failures),
        "qc_failed_series": len(failures),
        "large_nodule_reader_annotations": sum(
            record.qc.large_nodule_annotation_count
            for record in records
        ),
        "series_meeting_luna16_slice_thickness_rule": sum(
            record.qc.meets_luna16_slice_thickness_rule
            for record in records
        ),
        "scanner_manufacturers": dict(sorted(manufacturers.items())),
    }


def write_audit_summary(
    records: tuple[LIDCManifestRecord, ...],
    destination: str | Path,
) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            build_audit_summary(records),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )