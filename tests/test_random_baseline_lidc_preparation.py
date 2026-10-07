"""Synthetic DICOM/XML exercise the actual Commit-02/03 preparation bridge."""

from pathlib import Path

import numpy as np
import pytest
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from medfm_adapt3d.baseline.artifacts import read_sealed, seal, write_json
from medfm_adapt3d.baseline.candidates import (
    INPUT_TRACE_FIELDS,
    CandidateIndex,
    prepare_lidc_candidates,
)
from medfm_adapt3d.baseline.protocol import LockedProtocol
from medfm_adapt3d.baseline.selection import RunAccess


def _source(root: Path) -> tuple[Path, Path]:
    dicom, xml = root / "dicom", root / "xml"
    dicom.mkdir()
    xml.mkdir()
    study, series = generate_uid(), generate_uid()
    sops = []
    for z in range(5):
        sop = generate_uid()
        sops.append(sop)
        meta = Dataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = sop
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.ImplementationClassUID = generate_uid()
        path = dicom / f"slice-{z}.dcm"
        d = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
        d.PatientID = "SYNTHETIC-LIDC-FIXTURE"
        d.StudyInstanceUID, d.SeriesInstanceUID, d.SOPInstanceUID = study, series, sop
        d.SOPClassUID = CTImageStorage
        d.Modality = "CT"
        d.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        d.ImagePositionPatient = [0, 0, z]
        d.PixelSpacing = [1, 1]
        d.SliceThickness = 1
        d.Rows = d.Columns = 25
        d.SamplesPerPixel = 1
        d.PhotometricInterpretation = "MONOCHROME2"
        d.BitsAllocated = d.BitsStored = 16
        d.HighBit = 15
        d.PixelRepresentation = 1
        d.RescaleSlope = 1
        d.RescaleIntercept = -1024
        d.RescaleType = "HU"
        pixels = np.full((25, 25), 224, dtype=np.int16)
        pixels[10:15, 10:15] = 1124
        d.PixelData = pixels.tobytes()
        d.save_as(path, enforce_file_format=True)
    sessions = []
    for reader in (1, 2):
        points = [(10, 10), (14, 10), (14, 14), (10, 14)]
        edges = "".join(
            f"<edgeMap><xCoord>{x}</xCoord><yCoord>{y}</yCoord></edgeMap>" for x, y in points
        )
        roi = f"<roi><imageZposition>2</imageZposition><imageSOP_UID>{sops[2]}</imageSOP_UID>"
        sessions.append(
            f"<readingSession><unblindedReadNodule><noduleID>N{reader}</noduleID>"
            "<characteristics><subtlety>4</subtlety></characteristics>"
            f"{roi}<inclusion>TRUE</inclusion>{edges}</roi>"
            "</unblindedReadNodule></readingSession>"
        )
    text = f"<LidcReadMessage><ResponseHeader><StudyInstanceUID>{study}</StudyInstanceUID>"
    text += f"<SeriesInstanceUid>{series}</SeriesInstanceUid></ResponseHeader>"
    (xml / "source.xml").write_text(text + "".join(sessions) + "</LidcReadMessage>")
    return dicom, xml


def test_preparation_uses_actual_uid_reader_hu_and_geometry_contracts(tmp_path: Path) -> None:
    dicom, xml = _source(tmp_path)
    output = tmp_path / "candidates"
    path = prepare_lidc_candidates(
        dicom_root=dicom,
        xml_root=xml,
        preprocessing_path=Path("configs/preprocessing/lidc_candidate.yaml"),
        output=output,
    )
    index = CandidateIndex(path)
    assert len(index.patients) == 1 and len(index.records) == 2
    assert {r.reader_slot for r in index.records} == {1, 2}
    for record in index.records:
        image, target = index.load(record.candidate_id)
        assert image.shape == target.shape == (1, 65, 65, 65)
        assert record.spacing_mm_xyz == (1, 1, 1)
        assert record.trace["source_hu_range"] == [-800, 100]
        assert "mask_geometry" not in record.trace
        qc = read_sealed(index.root / record.target_qc_path)
        assert qc["trace"]["mask_geometry"]["passed"] is True
        assert qc["trace"]["mask_geometry"]["crop_volume_retention"] == 1
    with pytest.raises(FileExistsError):
        prepare_lidc_candidates(
            dicom_root=dicom,
            xml_root=xml,
            preprocessing_path=Path("configs/preprocessing/lidc_candidate.yaml"),
            output=output,
        )


def test_candidate_geometry_tampering_is_rejected(
    candidate_index: CandidateIndex, tmp_path: Path
) -> None:
    payload = read_sealed(candidate_index.path)
    payload["records"][0]["spacing_mm_xyz"] = [2, 1, 1]
    path = tmp_path / "changed-index.json"
    write_json(path, seal(payload))
    with pytest.raises(ValueError, match="spacing disagrees"):
        CandidateIndex(path)


def test_candidate_path_escape_is_rejected(candidate_index: CandidateIndex, tmp_path: Path) -> None:
    payload = read_sealed(candidate_index.path)
    payload["records"][0]["image_path"] = "../../outside.npy"
    path = tmp_path / "changed-index.json"
    write_json(path, seal(payload))
    with pytest.raises(ValueError, match="remain within"):
        CandidateIndex(path)


def test_input_metadata_contains_only_ct_grid_and_declared_oracle_centre(
    locked_protocol: LockedProtocol,
) -> None:
    for identifier in locked_protocol.partition_ids("test"):
        record = locked_protocol.index.by_id[identifier]
        assert set(record.trace) == set(INPUT_TRACE_FIELDS)
        assert record.trace["candidate_center_mm_lps"] == [0, 0, 0]
        assert "mask_geometry" not in record.trace
        assert "mask_interpolation" not in record.trace
        assert set(record.trace["output_geometry"]) == {
            "size_xyz",
            "spacing_mm_xyz",
            "origin_mm_lps",
            "direction_lps",
        }
        assert record.target_qc_path and len(record.target_qc_sha256) == 64


def test_test_target_qc_is_not_parsed_by_training_or_calibration(
    locked_protocol: LockedProtocol,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Path] = []
    original = read_sealed

    def track_qc(path: Path):
        seen.append(path)
        return original(path)

    monkeypatch.setattr("medfm_adapt3d.baseline.candidates.read_sealed", track_qc)
    for phase in ("calibration", "smoke", "definitive"):
        access = RunAccess(locked_protocol, 1, phase)
        test_id = access.allowed["test"][0]
        with pytest.raises(RuntimeError, match="sealed"):
            access.read("test", test_id)
        with pytest.raises(ValueError, match="authorized"):
            access.read("train", test_id)
        assert not seen
    access.read("train", access.allowed["train"][0])
    assert len(seen) == 1  # Authorized support QC is still verified, not discarded.
    assert seen[0] not in {
        locked_protocol.index.root / locked_protocol.index.by_id[i].target_qc_path
        for i in access.allowed["test"]
    }


def test_opaque_integrity_check_needs_no_label_or_qc_decoding(
    locked_protocol: LockedProtocol,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object):
        raise AssertionError("Opaque integrity checking must not decode targets or QC.")

    monkeypatch.setattr("medfm_adapt3d.baseline.candidates.read_sealed", forbidden)
    monkeypatch.setattr("medfm_adapt3d.baseline.candidates.np.load", forbidden)
    for identifier in locked_protocol.partition_ids("test"):
        locked_protocol.index.verify_integrity(identifier)


@pytest.mark.parametrize("location", ["trace", "record", "grid", "source_audit"])
def test_resealed_target_shape_injection_into_input_index_is_rejected(
    candidate_index: CandidateIndex,
    tmp_path: Path,
    location: str,
) -> None:
    payload = read_sealed(candidate_index.path)
    record = payload["records"][0]
    places = {
        "trace": record["trace"],
        "record": record,
        "grid": record["trace"]["output_geometry"],
        "source_audit": payload["source_audit"],
    }
    places[location]["mask_volume_mm3"] = 100.0
    path = tmp_path / "injected.json"
    write_json(path, seal(payload))
    with pytest.raises(ValueError, match="target|unexpected"):
        CandidateIndex(path)
