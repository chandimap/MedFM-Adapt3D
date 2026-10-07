import numpy as np
import pytest

from medfm_adapt3d.baseline.metrics import (
    ObservationMetrics,
    aggregate_patients,
    seed_statistics,
    segmentation_metrics,
)


def _metric(prediction: np.ndarray, reference: np.ndarray, spacing: tuple[float, ...] = (1, 1, 1)):
    return segmentation_metrics(
        prediction, reference, spacing_mm_xyz=spacing, patient_id="synthetic", candidate_id="target"
    )


@pytest.mark.parametrize("axis,distance", [(0, 4.0), (1, 3.0), (2, 2.0)])
def test_hd95_uses_xyz_spacing_for_zyx_arrays(axis: int, distance: float) -> None:
    reference = np.zeros((7, 7, 7), dtype=np.uint8)
    reference[3, 3, 3] = 1
    prediction = np.roll(reference, 1, axis=axis)
    assert _metric(prediction, reference, (2, 3, 4)).hd95_mm == pytest.approx(distance)


def test_physical_volume_error_and_voxel_sensitivity() -> None:
    reference = np.zeros((7, 7, 7), dtype=np.uint8)
    reference[2:4, 2:4, 2:4] = 1
    prediction = reference.copy()
    prediction[2, 2, 2] = 0
    result = _metric(prediction, reference, (2, 3, 4))
    assert result.absolute_volume_error_mm3 == 24.0
    assert result.voxel_sensitivity == 7 / 8
    assert result.dice == 14 / 15


def test_empty_prediction_is_an_explicit_failure() -> None:
    reference = np.ones((3, 3, 3), dtype=np.uint8)
    result = _metric(np.zeros_like(reference), reference, (0.5, 1, 2))
    assert result.dice == result.voxel_sensitivity == 0
    assert result.empty_prediction
    assert result.hd95_mm is None
    assert result.absolute_volume_error_mm3 == 27
    aggregate = aggregate_patients([result])
    assert aggregate["patient_macro"]["hd95_mm"] is None
    assert aggregate["counts"]["undefined_hd95_targets"] == 1
    assert aggregate["counts"]["empty_predictions"] == 1


def test_identical_masks_have_zero_hd95_even_at_crop_boundary() -> None:
    mask = np.ones((3, 3, 3), dtype=np.uint8)
    result = _metric(mask, mask)
    assert result.hd95_mm == 0
    assert result.dice == result.voxel_sensitivity == 1


def test_aggregation_gives_each_patient_equal_weight() -> None:
    rows = [ObservationMetrics("patient_a", f"a{i}", 1, 0, 0, 1, False, 1, 1) for i in range(4)]
    rows.append(ObservationMetrics("patient_b", "b", 0, None, 1, 0, True, 1, 0))
    result = aggregate_patients(rows)
    assert result["patient_macro"]["dice"] == 0.5
    assert result["patient_macro"]["empty_prediction_rate"] == 0.5
    assert result["patient_macro"]["hd95_mm"] is None
    assert result["hd95_mm_defined_patients_only"] == 0
    assert result["counts"]["fully_defined_hd95_patients"] == 1


def test_surface_metrics_are_not_cherry_picked_within_patient() -> None:
    rows = [
        ObservationMetrics("patient", "a", 1, 0, 0, 1, False, 1, 1),
        ObservationMetrics("patient", "b", 0, None, 1, 0, True, 1, 0),
    ]
    result = aggregate_patients(rows)
    assert result["patient_macro"]["hd95_mm"] is None
    assert result["patients"][0]["hd95_mm_defined_targets_only"] == 0
    assert result["counts"]["undefined_hd95_targets"] == 1


def test_seed_summary_keeps_every_seed_and_sample_sd() -> None:
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    result = seed_statistics(values)
    assert result["values"] == values
    assert result["mean"] == pytest.approx(0.3)
    assert result["std"] == pytest.approx(np.std(values, ddof=1))
    assert result["median"] == 0.3
    assert result["iqr"] == pytest.approx(0.2)
    assert result["min"] == 0.1 and result["max"] == 0.5
    partial = seed_statistics([None, 0.2])
    assert partial["conditional_on_defined_values"]
    assert partial["std"] is None and partial["n_undefined"] == 1
    assert partial["mean"] is None
    assert partial["defined_values_statistics"]["mean"] == 0.2


def test_empty_reference_or_invalid_spacing_rejected() -> None:
    mask = np.zeros((3, 3, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="foreground"):
        _metric(mask, mask)
    mask[1, 1, 1] = 1
    with pytest.raises(ValueError, match="spacing"):
        _metric(mask, mask, (0, 1, 1))
