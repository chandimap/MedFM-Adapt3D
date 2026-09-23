from pathlib import Path

import numpy as np
import pytest
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from medfm_adapt3d.preprocessing.config import DICOMValidationPolicy
from medfm_adapt3d.preprocessing.dicom import (
    DICOMValidationError,
    load_ct_dicom_series,
    rescale_stored_pixels_to_hu,
)


def _write_ct_slice(
    path: Path,
    *,
    study_uid: str,
    series_uid: str,
    z_position_mm: float,
    stored_value: int,
    sop_uid: str,
    include_rescale: bool = True,
) -> None:
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = CTImageStorage
    dataset.SOPInstanceUID = sop_uid
    dataset.PatientID = "SYNTHETIC-PATIENT"
    dataset.StudyInstanceUID = study_uid
    dataset.SeriesInstanceUID = series_uid
    dataset.Modality = "CT"
    dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    dataset.ImagePositionPatient = [0, 0, z_position_mm]
    dataset.PixelSpacing = [0.7, 0.8]
    dataset.SliceThickness = 2.5
    dataset.Rows = 4
    dataset.Columns = 4
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 1

    if include_rescale:
        dataset.RescaleSlope = 1.0
        dataset.RescaleIntercept = -1024.0
        dataset.RescaleType = "HU"

    pixels = np.full((4, 4), stored_value, dtype=np.int16)
    pixels[0, 0] = 1200
    dataset.PixelData = pixels.tobytes()
    dataset.save_as(path, enforce_file_format=True)


def _write_series(
    root: Path,
    *,
    z_positions: tuple[float, ...] = (5.0, 0.0, 2.5),
    include_rescale: bool = True,
) -> dict[float, str]:
    study_uid = generate_uid()
    series_uid = generate_uid()
    sop_uids_by_z: dict[float, str] = {}

    for index, z_position in enumerate(z_positions):
        sop_uid = generate_uid()
        sop_uids_by_z[z_position] = sop_uid
        _write_ct_slice(
            root / f"slice-{index}.dcm",
            study_uid=study_uid,
            series_uid=series_uid,
            z_position_mm=z_position,
            stored_value=index * 100,
            sop_uid=sop_uid,
            include_rescale=include_rescale,
        )

    return sop_uids_by_z


def test_dicom_rescale_is_applied_explicitly_in_float32() -> None:
    stored = np.asarray([[0, 100], [200, 300]], dtype=np.int16)

    hu = rescale_stored_pixels_to_hu(
        stored,
        slope=2.0,
        intercept=-1024.0,
    )

    assert hu.dtype == np.float32
    np.testing.assert_allclose(
        hu,
        np.asarray([[-1024.0, -824.0], [-624.0, -424.0]], dtype=np.float32),
    )


def test_dicom_series_loader_sorts_slices_and_preserves_patient_geometry(
    tmp_path: Path,
) -> None:
    sop_uids_by_z = _write_series(tmp_path)

    image, audit = load_ct_dicom_series(tmp_path)

    assert image.GetSize() == (4, 4, 3)
    assert image.GetSpacing() == pytest.approx((0.8, 0.7, 2.5))
    assert image.GetOrigin() == pytest.approx((0.0, 0.0, 0.0))
    assert image.GetDirection() == pytest.approx((1, 0, 0, 0, 1, 0, 0, 0, 1))
    assert audit.number_of_slices == 3
    assert audit.ordered_sop_instance_uids == tuple(
        sop_uids_by_z[position] for position in sorted(sop_uids_by_z)
    )
    assert audit.rescale_slopes == (1.0,)
    assert audit.rescale_intercepts == (-1024.0,)
    assert audit.minimum_hu == pytest.approx(-1024.0)
    assert audit.maximum_hu == pytest.approx(176.0)


def test_dicom_series_rejects_missing_explicit_hu_rescale(tmp_path: Path) -> None:
    _write_series(tmp_path, include_rescale=False)

    with pytest.raises(DICOMValidationError, match="explicit RescaleSlope"):
        load_ct_dicom_series(tmp_path)


def test_dicom_series_rejects_irregular_slice_spacing(tmp_path: Path) -> None:
    _write_series(tmp_path, z_positions=(0.0, 2.5, 5.2))

    with pytest.raises(DICOMValidationError, match="Irregular slice spacing"):
        load_ct_dicom_series(
            tmp_path,
            policy=DICOMValidationPolicy(
                maximum_slice_spacing_deviation_mm=0.05,
            ),
        )
