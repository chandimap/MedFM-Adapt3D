import pytest

from medfm_adapt3d.data.lidc_manifest import build_manifest_record
from medfm_adapt3d.data.lidc_schema import (
    DICOMSeriesSummary,
    NoduleCharacteristics,
    ReaderNoduleAnnotation,
    ROIReference,
)
from medfm_adapt3d.data.lidc_xml import LIDCXMLScan


def _dicom_summary() -> DICOMSeriesSummary:
    return DICOMSeriesSummary(
        patient_id="LIDC-IDRI-0001",
        study_instance_uid="1.2.3",
        series_instance_uid="4.5.6",
        modality="CT",
        sop_instance_uids=(
            "1.2.840.1",
            "1.2.840.2",
        ),
        rows=512,
        columns=512,
        pixel_spacing_mm=(0.7, 0.7),
        slice_thickness_mm=1.25,
        z_positions_mm=(-101.5, -100.0),
        manufacturer="Example",
        manufacturer_model_name="ResearchScanner",
        convolution_kernel="STANDARD",
        kvp=120.0,
        source_directory="synthetic",
    )


def test_manifest_passes_when_annotation_sop_is_present() -> None:
    annotation = ReaderNoduleAnnotation(
        reader_slot=1,
        reader_annotation_id="N1",
        characteristics=NoduleCharacteristics(
            subtlety=4,
            malignancy=3,
        ),
        rois=(
            ROIReference(
                sop_instance_uid="1.2.840.1",
                z_position_mm=-101.5,
                inclusion=True,
                edge_points=((10, 20),),
            ),
        ),
    )

    xml = LIDCXMLScan(
        study_instance_uid="1.2.3",
        series_instance_uid="4.5.6",
        annotations=(annotation,),
        source_xml="synthetic.xml",
    )

    record = build_manifest_record(
        _dicom_summary(),
        xml,
    )

    assert record.qc.passed
    assert record.qc.annotation_count == 1
    assert record.qc.large_nodule_annotation_count == 1
    assert record.qc.missing_annotation_sop_reference_count == 0
    assert record.qc.meets_luna16_slice_thickness_rule


def test_manifest_fails_when_annotation_references_missing_slice() -> None:
    annotation = ReaderNoduleAnnotation(
        reader_slot=1,
        reader_annotation_id="N1",
        characteristics=NoduleCharacteristics(malignancy=3),
        rois=(
            ROIReference(
                sop_instance_uid="missing.sop.uid",
                z_position_mm=-101.5,
                inclusion=True,
                edge_points=((10, 20),),
            ),
        ),
    )

    xml = LIDCXMLScan(
        study_instance_uid="1.2.3",
        series_instance_uid="4.5.6",
        annotations=(annotation,),
        source_xml="synthetic.xml",
    )

    record = build_manifest_record(
        _dicom_summary(),
        xml,
    )

    assert not record.qc.passed
    assert (
        "annotation_sop_reference_missing_from_dicom_series"
        in record.qc.issues
    )
    assert record.qc.missing_annotation_sop_reference_count == 1


def test_luna16_thickness_rule_is_recorded_not_used_as_base_exclusion() -> None:
    dicom = DICOMSeriesSummary(
        patient_id="LIDC-IDRI-0002",
        study_instance_uid="9.8.7",
        series_instance_uid="6.5.4",
        modality="CT",
        sop_instance_uids=(
            "9.9.1",
            "9.9.2",
        ),
        rows=512,
        columns=512,
        pixel_spacing_mm=(0.8, 0.8),
        slice_thickness_mm=3.0,
        z_positions_mm=(0.0, 3.0),
        manufacturer=None,
        manufacturer_model_name=None,
        convolution_kernel=None,
        kvp=None,
        source_directory="synthetic",
    )

    xml = LIDCXMLScan(
        study_instance_uid="9.8.7",
        series_instance_uid="6.5.4",
        annotations=(),
        source_xml="synthetic.xml",
    )

    record = build_manifest_record(
        dicom,
        xml,
    )

    assert record.qc.passed
    assert not record.qc.meets_luna16_slice_thickness_rule

def test_manifest_rejects_dicom_xml_series_mismatch() -> None:
    xml = LIDCXMLScan(
        study_instance_uid="1.2.3",
        series_instance_uid="DIFFERENT-SERIES",
        annotations=(),
        source_xml="synthetic.xml",
    )

    with pytest.raises(
        ValueError,
        match="SeriesInstanceUID mismatch",
    ):
        build_manifest_record(
            _dicom_summary(),
            xml,
        )