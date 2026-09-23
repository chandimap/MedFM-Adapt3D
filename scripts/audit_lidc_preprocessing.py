"""Audit Geometry Preservation for Every Reader Nodule in one Local LIDC Series."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from medfm_adapt3d.data.lidc_xml import parse_lidc_xml
from medfm_adapt3d.preprocessing import (
    CandidatePreprocessor,
    load_ct_dicom_series,
    load_preprocessing_protocol,
    rasterize_reader_nodule,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the version-controlled preprocessing contract on all >=3 mm "
            "reader annotations in one local LIDC-IDRI CT series."
        )
    )
    parser.add_argument(
        "--dicom-series",
        type=Path,
        required=True,
        help="Directory containing one CT DICOM series.",
    )
    parser.add_argument(
        "--xml",
        type=Path,
        required=True,
        help="LIDC XML file linked to the DICOM series.",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/preprocessing/lidc_candidate.yaml"),
        help="Version-controlled preprocessing protocol.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON destination; otherwise print to standard output.",
    )
    return parser.parse_args()


def build_preprocessing_audit(
    *,
    dicom_series: Path,
    xml_path: Path,
    protocol_path: Path,
) -> dict[str, object]:
    protocol = load_preprocessing_protocol(protocol_path)
    image, dicom_audit = load_ct_dicom_series(
        dicom_series,
        policy=protocol.dicom,
    )
    xml_scan = parse_lidc_xml(xml_path)

    if xml_scan.study_instance_uid != dicom_audit.study_instance_uid:
        raise ValueError("DICOM/XML StudyInstanceUID mismatch.")
    if xml_scan.series_instance_uid != dicom_audit.series_instance_uid:
        raise ValueError("DICOM/XML SeriesInstanceUID mismatch.")

    annotations = sorted(
        (
            annotation
            for annotation in xml_scan.annotations
            if annotation.is_large_nodule
        ),
        key=lambda annotation: (
            annotation.reader_slot,
            annotation.reader_annotation_id,
        ),
    )
    if not annotations:
        raise ValueError("No >=3 mm reader nodule annotations were found.")

    preprocessor = CandidatePreprocessor(
        protocol.preprocessing,
        protocol.geometry_qc,
    )
    annotation_audits: list[dict[str, object]] = []
    all_geometry_checks_passed = True

    for annotation in annotations:
        rasterized = rasterize_reader_nodule(
            annotation,
            source_image=image,
            ordered_sop_instance_uids=dicom_audit.ordered_sop_instance_uids,
        )
        candidate = preprocessor(
            image,
            candidate_center_mm_lps=rasterized.candidate_center_mm_lps,
            mask=rasterized.mask,
        )
        mask_geometry = candidate.trace.mask_geometry
        if mask_geometry is None:
            raise RuntimeError("A rasterized reader mask produced no geometry audit.")
        all_geometry_checks_passed &= mask_geometry.passed
        annotation_audits.append(
            {
                "reader_slot": rasterized.reader_slot,
                "reader_annotation_id": rasterized.reader_annotation_id,
                "trace": candidate.trace.to_dict(),
                "tensor_shape_cdhw": list(candidate.image.shape),
            }
        )

    return {
        "schema_version": protocol.schema_version,
        "dicom": asdict(dicom_audit),
        "reader_nodule_annotations": annotation_audits,
        "all_geometry_checks_passed": all_geometry_checks_passed,
        "clinical_performance_claimed": False,
    }


def main() -> None:
    arguments = _arguments()
    payload = build_preprocessing_audit(
        dicom_series=arguments.dicom_series,
        xml_path=arguments.xml,
        protocol_path=arguments.protocol,
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"

    if arguments.output is None:
        print(rendered, end="")
        return

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(rendered, encoding="utf-8")
    print(f"Wrote preprocessing audit: {arguments.output}")


if __name__ == "__main__":
    main()
