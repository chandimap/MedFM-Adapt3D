"""Patient-macro Overlap and Physical Metrics with Explicit Segmentation Failures."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import SimpleITK as sitk

from medfm_adapt3d.baseline.config import METRICS


@dataclass(frozen=True, slots=True)
class ObservationMetrics:
    patient_id: str
    candidate_id: str
    dice: float
    hd95_mm: float | None
    absolute_volume_error_mm3: float
    voxel_sensitivity: float
    empty_prediction: bool
    reference_voxels: int
    predicted_voxels: int


def _surface_distances(mask: np.ndarray, other: np.ndarray, spacing: Sequence[float]) -> np.ndarray:
    # Padding with background includes surfaces touching the candidate boundary.
    images = [sitk.GetImageFromArray(np.pad(a.astype(np.uint8), 1)) for a in (mask, other)]
    for image in images:
        image.SetSpacing(tuple(float(v) for v in spacing))
    surfaces = [sitk.BinaryContour(image, fullyConnected=False) for image in images]
    distance = sitk.Abs(
        sitk.SignedMaurerDistanceMap(
            surfaces[1],
            insideIsPositive=False,
            squaredDistance=False,
            useImageSpacing=True,
        )
    )
    return sitk.GetArrayFromImage(distance)[sitk.GetArrayFromImage(surfaces[0]) > 0]


def segmentation_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    spacing_mm_xyz: Sequence[float],
    patient_id: str,
    candidate_id: str,
) -> ObservationMetrics:
    """HD95 is max of the two directed 95th percentiles, in physical millimetres.

    References must be non-empty reader targets. Empty predictions have Dice
    and sensitivity zero and HD95 undefined (None), never an invented distance.
    """
    prediction = np.asarray(prediction)
    reference = np.asarray(reference)
    if prediction.ndim != 3 or prediction.shape != reference.shape:
        raise ValueError("Metric inputs must be aligned [D,H,W] binary volumes.")
    if not np.isin(prediction, [0, 1]).all() or not np.isin(reference, [0, 1]).all():
        raise ValueError("Metric inputs must be binary; probabilities need the frozen threshold.")
    if len(spacing_mm_xyz) != 3 or not all(np.isfinite(v) and v > 0 for v in spacing_mm_xyz):
        raise ValueError("Physical metrics require three finite positive voxel spacings in XYZ.")
    pred, ref = prediction.astype(bool), reference.astype(bool)
    n_pred, n_ref = int(pred.sum()), int(ref.sum())
    if not n_ref:
        raise ValueError("Oracle-centred reference masks must contain foreground voxels.")
    overlap = int((pred & ref).sum())
    hd95 = None
    if n_pred:
        hd95 = max(
            float(np.percentile(_surface_distances(a, b, spacing_mm_xyz), 95))
            for a, b in ((pred, ref), (ref, pred))
        )
    return ObservationMetrics(
        patient_id,
        candidate_id,
        2.0 * overlap / (n_pred + n_ref),
        hd95,
        abs(n_pred - n_ref) * float(np.prod(spacing_mm_xyz)),
        overlap / n_ref,
        n_pred == 0,
        n_ref,
        n_pred,
    )


def aggregate_patients(observations: Sequence[ObservationMetrics]) -> dict[str, Any]:
    """Reader/candidate mean within patient, then an equally weighted patient mean.

    Primary HD95 remains undefined if ANY observation is undefined. A separate
    conditional HD95 is labelled explicitly and accompanied by failure counts;
    exclusion of empty predictions cannot manufacture a favourable primary HD95.
    """
    if not observations or len({o.candidate_id for o in observations}) != len(observations):
        raise ValueError("Patient aggregation requires non-empty, unique candidate observations.")
    grouped: dict[str, list[ObservationMetrics]] = defaultdict(list)
    for observation in observations:
        grouped[observation.patient_id].append(observation)
    patients = []
    for identifier, rows in sorted(grouped.items()):
        values: dict[str, Any] = {"patient_id": identifier, "reader_target_count": len(rows)}
        for metric in METRICS:
            data = [getattr(row, metric) for row in rows]
            values[metric] = None if any(v is None for v in data) else float(np.mean(data))
        defined = [r.hd95_mm for r in rows if r.hd95_mm is not None]
        values.update(
            hd95_mm_defined_targets_only=float(np.mean(defined)) if defined else None,
            undefined_hd95_count=len(rows) - len(defined),
            empty_prediction_count=sum(r.empty_prediction for r in rows),
            empty_prediction_rate=sum(r.empty_prediction for r in rows) / len(rows),
        )
        patients.append(values)
    macro: dict[str, Any] = {}
    for metric in (*METRICS, "empty_prediction_rate"):
        data = [p[metric] for p in patients]
        macro[metric] = None if any(v is None for v in data) else float(np.mean(data))
    defined_patients = [p["hd95_mm"] for p in patients if p["hd95_mm"] is not None]
    return {
        "patient_macro": macro,
        "patients": patients,
        "observations": [asdict(o) for o in observations],
        "hd95_mm_defined_patients_only": (
            float(np.mean(defined_patients)) if defined_patients else None
        ),
        "counts": {
            "patients": len(patients),
            "reader_targets": len(observations),
            "empty_predictions": sum(o.empty_prediction for o in observations),
            "undefined_hd95_targets": sum(o.hd95_mm is None for o in observations),
            "fully_defined_hd95_patients": len(defined_patients),
        },
    }


def seed_statistics(values: Sequence[float | None]) -> dict[str, Any]:
    """Descriptive seed spread, with completeness and sample SD; no patient CI."""
    defined = np.asarray([v for v in values if v is not None], dtype=np.float64)
    result: dict[str, Any] = {
        "values": list(values),
        "n_defined": len(defined),
        "n_undefined": len(values) - len(defined),
    }
    for name in ("mean", "std", "median", "iqr", "min", "max"):
        result[name] = None
    if defined.size:
        q25, q75 = np.percentile(defined, [25, 75])
        result.update(
            mean=float(defined.mean()),
            median=float(np.median(defined)),
            iqr=float(q75 - q25),
            min=float(defined.min()),
            max=float(defined.max()),
            std=float(defined.std(ddof=1)) if len(defined) >= 2 else None,
        )
    result["conditional_on_defined_values"] = result["n_undefined"] > 0
    if result["n_undefined"]:
        names = ("mean", "std", "median", "iqr", "min", "max")
        result["defined_values_statistics"] = {name: result[name] for name in names}
        for name in names:
            result[name] = None
    return result
