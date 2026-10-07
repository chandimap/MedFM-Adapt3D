"""Private, Integrity-checked Reader Targets Built through Commits 02–03."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from medfm_adapt3d.baseline.artifacts import (
    file_hash,
    fingerprint,
    local_path,
    read_sealed,
    seal,
    write_json,
)
from medfm_adapt3d.baseline.config import TASK
from medfm_adapt3d.data.dicom_index import index_dicom_series
from medfm_adapt3d.data.lidc_manifest import build_lidc_manifest
from medfm_adapt3d.preprocessing import (
    CandidatePreprocessor,
    PreprocessedCandidate,
    load_ct_dicom_series,
    load_preprocessing_protocol,
    rasterize_reader_nodule,
)

INPUT_TRACE_FIELDS = (
    "source_geometry",
    "output_geometry",
    "source_hu_range",
    "output_intensity_range",
    "candidate_center_mm_lps",
    "crop_centering_error_mm",
    "image_interpolation",
)


def input_trace(trace: dict[str, Any]) -> dict[str, Any]:
    """Project CT geometry plus the declared oracle centre, never mask-shape QC."""
    return {name: trace[name] for name in INPUT_TRACE_FIELDS}


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    """Input-only metadata for one reader observation, with opaque target references.

    The oracle centre is part of the task. Mask measurements and geometry QC
    remain in a separately sealed file, opened only with authorized target loads.
    """

    candidate_id: str
    patient_id: str
    study_uid: str
    series_uid: str
    reader_slot: int
    annotation_id: str
    image_path: str
    mask_path: str
    image_sha256: str
    mask_sha256: str
    target_qc_path: str
    target_qc_sha256: str
    shape_cdhw: tuple[int, ...]
    spacing_mm_xyz: tuple[float, ...]
    trace: dict[str, Any]


def candidate_identity(patient: str, study: str, series: str, reader: int, target: str) -> str:
    return fingerprint([patient, study, series, reader, target])


def save_candidate(
    root: Path,
    candidate: PreprocessedCandidate,
    *,
    patient_id: str,
    study_uid: str,
    series_uid: str,
    reader_slot: int,
    annotation_id: str,
) -> CandidateRecord:
    """Saving image and target separately so a test target can stay unopened until lock."""
    if candidate.mask is None or candidate.trace.mask_geometry is None:
        raise ValueError("Baseline candidates require a geometry-audited reader target.")
    if not candidate.trace.mask_geometry.passed:
        raise ValueError("Failed Commit-03 geometry QC cannot enter baseline training.")
    identifier = candidate_identity(patient_id, study_uid, series_uid, reader_slot, annotation_id)
    paths = (f"arrays/{identifier}_image.npy", f"arrays/{identifier}_mask.npy")
    for relative, tensor in zip(paths, (candidate.image, candidate.mask), strict=True):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            np.save(stream, tensor.cpu().numpy(), allow_pickle=False)
    image_digest, mask_digest = (file_hash(root / path) for path in paths)
    qc_path = f"target_qc/{identifier}.json"
    write_json(
        root / qc_path,
        seal(
            {
                "schema_version": "1.0",
                "candidate_id": identifier,
                "image_sha256": image_digest,
                "mask_sha256": mask_digest,
                "trace": candidate.trace.to_dict(),
            }
        ),
    )
    return CandidateRecord(
        candidate_id=identifier,
        patient_id=patient_id,
        study_uid=study_uid,
        series_uid=series_uid,
        reader_slot=reader_slot,
        annotation_id=annotation_id,
        image_path=paths[0],
        mask_path=paths[1],
        image_sha256=image_digest,
        mask_sha256=mask_digest,
        target_qc_path=qc_path,
        target_qc_sha256=file_hash(root / qc_path),
        shape_cdhw=tuple(candidate.image.shape),
        spacing_mm_xyz=candidate.trace.output_geometry.spacing_mm_xyz,
        trace=input_trace(candidate.trace.to_dict()),
    )


def write_candidate_index(
    root: Path,
    records: list[CandidateRecord],
    *,
    preprocessing_path: Path,
    source_audit: dict[str, Any],
    synthetic: bool = False,
) -> Path:
    protocol = load_preprocessing_protocol(preprocessing_path)
    if (
        not protocol.preprocessing.fail_on_geometry_qc
        or not protocol.dicom.require_explicit_rescale
    ):
        raise ValueError("Baseline preparation requires all Commit-03 fail-closed QC gates.")
    if not records:
        raise ValueError(
            "No eligible geometry-passing reader targets; candidate preparation failed."
        )
    path = root / "index.json"
    payload = {
        "schema_version": "1.1",
        "task": TASK,
        "synthetic": synthetic,
        "preprocessing_sha256": fingerprint(asdict(protocol)),
        "preprocessing_file_sha256": file_hash(preprocessing_path),
        "preprocessing_protocol": asdict(protocol),
        "source_audit": source_audit,
        "records": [asdict(r) for r in sorted(records, key=lambda r: r.candidate_id)],
        "observation_unit": "reader-specific oracle-centred candidate target",
        "reader_matching": "No cross-reader lesion matching or consensus is inferred.",
    }
    write_json(path, seal(payload))
    CandidateIndex(path)  # Validate metadata before declaring preparation complete.
    return path


class CandidateIndex:
    """Input metadata excludes target shape; authorized loads inspect sealed QC.

    This is a project API boundary, not encryption or an OS sandbox. The
    researcher can access private preparation files directly outside this API.
    """

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.root = self.path.parent
        self.payload = read_sealed(path)
        self.sha256 = fingerprint(self.payload)
        if self.payload.get("schema_version") != "1.1" or self.payload.get("task") != TASK:
            raise ValueError("Unsupported candidate index schema/task.")
        if set(self.payload) != {
            "schema_version",
            "task",
            "synthetic",
            "preprocessing_sha256",
            "preprocessing_file_sha256",
            "preprocessing_protocol",
            "source_audit",
            "records",
            "observation_unit",
            "reader_matching",
        }:
            raise ValueError("Candidate index contains unexpected metadata fields.")
        audit = self.payload["source_audit"]
        if set(audit) - {
            "manifest_sha256",
            "reconciled_series",
            "indexed_ct_series",
            "unmatched_series",
            "excluded_series",
            "evidence_source",
        }:
            raise ValueError("Source audit contains unexpected target-derived metadata.")
        for item in audit.get("unmatched_series", []):
            if set(item) != {"study_uid", "series_uid", "patient_id", "reason"}:
                raise ValueError("Unmatched source audit contains unexpected metadata.")
        for item in audit.get("excluded_series", []):
            if set(item) != {"series_uid", "reason"}:
                raise ValueError("Excluded source audit contains unexpected metadata.")
        protocol = self.payload["preprocessing_protocol"]
        if fingerprint(protocol) != self.payload["preprocessing_sha256"]:
            raise ValueError("Preprocessing protocol fingerprint mismatch.")
        if protocol["preprocessing"]["fail_on_geometry_qc"] is not True:
            raise ValueError("Preprocessing QC must remain fail-closed.")
        if any(
            set(item) != {f.name for f in fields(CandidateRecord)}
            for item in self.payload["records"]
        ):
            raise ValueError("Candidate records contain unexpected target-shape metadata fields.")
        self.records = tuple(
            CandidateRecord(
                **{
                    **item,
                    "shape_cdhw": tuple(item["shape_cdhw"]),
                    "spacing_mm_xyz": tuple(item["spacing_mm_xyz"]),
                }
            )
            for item in self.payload["records"]
        )
        self.by_id = {r.candidate_id: r for r in self.records}
        if not self.records or len(self.by_id) != len(self.records):
            raise ValueError("Candidate index is empty or contains duplicated observations.")
        owners: dict[tuple[str, str], str] = {}
        for r in self.records:
            identity = candidate_identity(
                r.patient_id,
                r.study_uid,
                r.series_uid,
                r.reader_slot,
                r.annotation_id,
            )
            if identity != r.candidate_id or not r.patient_id.strip():
                raise ValueError("Tampered candidate lineage/identity.")
            if not r.annotation_id or type(r.reader_slot) is not int or r.reader_slot < 1:
                raise ValueError("Missing scan-local reader identity.")
            series = (r.study_uid, r.series_uid)
            if series in owners and owners[series] != r.patient_id:
                raise ValueError("The same source CT series cannot belong to two patients.")
            owners[series] = r.patient_id
            self._validate_geometry(r, protocol)
            for relative in (r.image_path, r.mask_path, r.target_qc_path):
                local_path(self.root, relative)

    @staticmethod
    def _validate_geometry(r: CandidateRecord, protocol: dict[str, Any]) -> None:
        if len(r.shape_cdhw) != 4 or r.shape_cdhw[0] != 1 or min(r.shape_cdhw) <= 0:
            raise ValueError("Candidate shape must be [1,D,H,W].")
        if len(r.spacing_mm_xyz) != 3 or not all(
            math.isfinite(v) and v > 0 for v in r.spacing_mm_xyz
        ):
            raise ValueError("Candidate spacing must be three positive physical values.")
        if set(r.trace) != set(INPUT_TRACE_FIELDS):
            raise ValueError("Input trace must exclude target-shape/QC metadata.")
        for name in ("source_geometry", "output_geometry"):
            if set(r.trace[name]) != {
                "size_xyz",
                "spacing_mm_xyz",
                "origin_mm_lps",
                "direction_lps",
            }:
                raise ValueError("CT grid contains unexpected target metadata.")
        centre = r.trace["candidate_center_mm_lps"]
        if len(centre) != 3 or not all(math.isfinite(v) for v in centre):
            raise ValueError("The declared oracle centre requires three finite LPS coordinates.")
        geometry = r.trace["output_geometry"]
        if tuple(geometry["size_xyz"]) != r.shape_cdhw[1:][::-1]:
            raise ValueError("Candidate tensor and physical grid dimensions diverged.")
        if tuple(geometry["spacing_mm_xyz"]) != r.spacing_mm_xyz:
            raise ValueError("Candidate spacing disagrees with preprocessing trace.")
        if tuple(protocol["preprocessing"]["target_spacing_mm"]) != r.spacing_mm_xyz:
            raise ValueError("Candidate spacing disagrees with the frozen preprocessing protocol.")
        if not np.allclose(geometry["direction_lps"], np.eye(3).ravel(), atol=1e-6):
            raise ValueError("Commit-03 axis-aligned LPS geometry is required.")

    @property
    def patients(self) -> tuple[str, ...]:
        return tuple(sorted({r.patient_id for r in self.records}))

    def verify_integrity(self, candidate_id: str) -> None:
        """Hash opaque array/QC bytes without parsing labels or target measurements."""
        r = self.by_id[candidate_id]
        for relative, digest in (
            (r.image_path, r.image_sha256),
            (r.mask_path, r.mask_sha256),
            (r.target_qc_path, r.target_qc_sha256),
        ):
            path = local_path(self.root, relative)
            if file_hash(path) != digest:
                raise ValueError(f"Candidate file integrity failure: {path.name}.")

    def load(self, candidate_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Low-level authorized target load; the training path must use RunAccess.

        Full Commit-03 QC is verified here, after RunAccess authorization, rather
        than published in the input index. Do not call this directly for test.
        """
        r = self.by_id[candidate_id]
        self.verify_integrity(candidate_id)
        qc = read_sealed(local_path(self.root, r.target_qc_path))
        if (
            qc.get("schema_version") != "1.0"
            or qc.get("candidate_id") != candidate_id
            or qc.get("image_sha256") != r.image_sha256
            or qc.get("mask_sha256") != r.mask_sha256
            or input_trace(qc["trace"]) != r.trace
        ):
            raise ValueError("Target QC no longer matches the candidate lineage/input grid.")
        audit = qc["trace"]["mask_geometry"]
        if not audit["passed"]:
            raise ValueError("Candidate mask failed Commit-03 geometry QC.")
        image = np.load(local_path(self.root, r.image_path), allow_pickle=False)
        mask = np.load(local_path(self.root, r.mask_path), allow_pickle=False)
        if image.shape != r.shape_cdhw or mask.shape != r.shape_cdhw:
            raise ValueError("Candidate arrays no longer match the aligned index geometry.")
        if image.dtype != np.float32 or mask.dtype != np.uint8:
            raise ValueError("Candidates require float32 CT and uint8 binary reader masks.")
        if not np.isfinite(image).all() or not np.isin(mask, [0, 1]).all() or not mask.any():
            raise ValueError("Candidate arrays contain non-finite CT or invalid/empty reference.")
        if audit["cropped"]["voxel_count"] != int(mask.sum()):
            raise ValueError("Authorized target foreground disagrees with its sealed geometry QC.")
        low, high = self.payload["preprocessing_protocol"]["preprocessing"]["output_range"]
        if image.min() < low - 1e-6 or image.max() > high + 1e-6:
            raise ValueError("Candidate intensities violate the preprocessing range.")
        return torch.from_numpy(image), torch.from_numpy(mask)


def prepare_lidc_candidates(
    *,
    dicom_root: Path,
    xml_root: Path,
    preprocessing_path: Path,
    output: Path,
) -> Path:
    """Using existing manifest reconciliation, HU loading, rasterization, and spatial QC.

    Geometry/QC failures abort preparation. Patients without eligible >=3 mm
    reader contours are explicitly recorded; they cannot count toward K.
    """
    output.mkdir(parents=True, exist_ok=False)
    protocol = load_preprocessing_protocol(preprocessing_path)
    manifest = build_lidc_manifest(dicom_root=dicom_root, xml_root=xml_root)
    if not manifest:
        raise ValueError("No exactly reconciled DICOM/XML CT series were found.")
    # Reuse the existing header index to make non-reconciled source series
    # explicit without changing Commit-02 reconciliation/duplicate handling.
    source_series = index_dicom_series(dicom_root)
    matched = {(r.study_instance_uid, r.series_instance_uid) for r in manifest}
    unmatched = [
        {
            "study_uid": study,
            "series_uid": series,
            "patient_id": scan.patient_id,
            "reason": "no_matching_xml_in_supplied_source",
        }
        for (study, series), scan in sorted(source_series.items())
        if (study, series) not in matched
    ]
    preprocessor = CandidatePreprocessor(protocol.preprocessing, protocol.geometry_qc)
    records: list[CandidateRecord] = []
    excluded: list[dict[str, str]] = []
    for scan in manifest:
        if not scan.qc.passed:
            raise ValueError(f"Cohort QC failed for a source series: {scan.qc.issues}.")
        annotations = [a for a in scan.annotations if a.is_large_nodule]
        if not annotations:
            excluded.append(
                {
                    "series_uid": scan.series_instance_uid,
                    "reason": "no_eligible_ge_3mm_reader_contours",
                }
            )
            continue
        image, dicom_audit = load_ct_dicom_series(
            scan.dicom.source_directory,
            series_instance_uid=scan.series_instance_uid,
            policy=protocol.dicom,
        )
        if dicom_audit.study_instance_uid != scan.study_instance_uid:
            raise ValueError("Source study identity changed after cohort reconciliation.")
        if set(dicom_audit.ordered_sop_instance_uids) != set(scan.dicom.sop_instance_uids):
            raise ValueError("Source SOP membership changed after cohort reconciliation.")
        for annotation in sorted(
            annotations, key=lambda a: (a.reader_slot, a.reader_annotation_id)
        ):
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
            records.append(
                save_candidate(
                    output,
                    candidate,
                    patient_id=scan.patient_id,
                    study_uid=scan.study_instance_uid,
                    series_uid=scan.series_instance_uid,
                    reader_slot=annotation.reader_slot,
                    annotation_id=annotation.reader_annotation_id,
                )
            )
    return write_candidate_index(
        output,
        records,
        preprocessing_path=preprocessing_path,
        source_audit={
            "manifest_sha256": fingerprint([r.to_dict() for r in manifest]),
            "reconciled_series": len(manifest),
            "indexed_ct_series": len(source_series),
            "unmatched_series": unmatched,
            "excluded_series": excluded,
        },
    )
