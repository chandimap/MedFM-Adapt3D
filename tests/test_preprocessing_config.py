from pathlib import Path

import pytest

from medfm_adapt3d.preprocessing.config import (
    GeometryPreservationCriteria,
    PreprocessingConfig,
    load_preprocessing_protocol,
)


def test_committed_preprocessing_protocol_is_complete_and_valid() -> None:
    protocol = load_preprocessing_protocol(
        Path("configs/preprocessing/lidc_candidate.yaml")
    )

    assert protocol.schema_version == "1.0"
    assert protocol.preprocessing.target_spacing_mm == (1.0, 1.0, 1.0)
    assert protocol.preprocessing.crop_size_voxels() == (65, 65, 65)
    assert protocol.preprocessing.hu_window == (-1000.0, 400.0)
    assert protocol.preprocessing.output_range == (-1.0, 1.0)
    assert protocol.geometry_qc.maximum_centroid_shift_mm == 1.5
    assert protocol.geometry_qc.tiny_nodule_maximum_source_volume_mm3 == 150.0


def test_crop_size_is_odd_and_covers_requested_physical_extent() -> None:
    config = PreprocessingConfig(
        target_spacing_mm=(0.7, 1.0, 1.5),
        crop_extent_mm=(32.0, 32.0, 32.0),
    )

    size = config.crop_size_voxels()

    assert all(value % 2 == 1 for value in size)
    assert all(
        (value - 1) * spacing >= extent
        for value, spacing, extent in zip(
            size,
            config.target_spacing_mm,
            config.crop_extent_mm,
            strict=True,
        )
    )


def test_mask_interpolation_cannot_be_changed_to_linear() -> None:
    with pytest.raises(ValueError, match="nearest-neighbour"):
        PreprocessingConfig(mask_interpolation="linear")


def test_geometry_qc_rejects_inverted_volume_bounds() -> None:
    with pytest.raises(ValueError, match="must not be smaller"):
        GeometryPreservationCriteria(
            minimum_resampled_volume_ratio=1.1,
            maximum_resampled_volume_ratio=0.9,
        )
