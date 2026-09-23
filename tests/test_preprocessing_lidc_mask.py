import numpy as np
import pytest
import SimpleITK as sitk

from medfm_adapt3d.data.lidc_schema import ReaderNoduleAnnotation, ROIReference
from medfm_adapt3d.preprocessing.geometry import assert_same_geometry
from medfm_adapt3d.preprocessing.lidc_mask import (
    LIDCContourError,
    rasterize_reader_nodule,
)


def _source_image() -> sitk.Image:
    image = sitk.Image((20, 20, 3), sitk.sitkFloat32)
    image.SetSpacing((0.8, 0.8, 1.5))
    image.SetOrigin((-8.0, -8.0, -1.5))
    return image


def _square(
    *,
    sop_uid: str,
    z_position_mm: float,
    minimum: int,
    maximum: int,
    inclusion: bool,
) -> ROIReference:
    return ROIReference(
        sop_instance_uid=sop_uid,
        z_position_mm=z_position_mm,
        inclusion=inclusion,
        edge_points=(
            (minimum, minimum),
            (maximum, minimum),
            (maximum, maximum),
            (minimum, maximum),
        ),
    )


def test_reader_contours_map_to_sop_linked_slices_and_preserve_holes() -> None:
    image = _source_image()
    sop_uids = ("SOP-0", "SOP-1", "SOP-2")
    annotation = ReaderNoduleAnnotation(
        reader_slot=2,
        reader_annotation_id="NODULE-7",
        characteristics=None,
        rois=(
            _square(
                sop_uid="SOP-1",
                z_position_mm=0.0,
                minimum=5,
                maximum=12,
                inclusion=True,
            ),
            _square(
                sop_uid="SOP-1",
                z_position_mm=0.0,
                minimum=8,
                maximum=9,
                inclusion=False,
            ),
            _square(
                sop_uid="SOP-2",
                z_position_mm=1.5,
                minimum=6,
                maximum=11,
                inclusion=True,
            ),
        ),
    )

    result = rasterize_reader_nodule(
        annotation,
        source_image=image,
        ordered_sop_instance_uids=sop_uids,
    )

    assert_same_geometry(image, result.mask)
    mask = sitk.GetArrayFromImage(result.mask)
    assert mask[0].sum() == 0
    assert mask[1, 8, 8] == 0
    assert mask[1].sum() > 0
    assert mask[2].sum() > 0
    assert set(np.unique(mask).tolist()) == {0, 1}
    assert result.reader_slot == 2
    assert result.reader_annotation_id == "NODULE-7"
    assert result.referenced_sop_instance_uids == ("SOP-1", "SOP-2")
    assert len(result.candidate_center_mm_lps) == 3


def test_reader_contour_rejects_unknown_sop_reference() -> None:
    image = _source_image()
    annotation = ReaderNoduleAnnotation(
        reader_slot=1,
        reader_annotation_id="NODULE-1",
        characteristics=None,
        rois=(
            _square(
                sop_uid="UNKNOWN",
                z_position_mm=0.0,
                minimum=5,
                maximum=10,
                inclusion=True,
            ),
        ),
    )

    with pytest.raises(LIDCContourError, match="absent"):
        rasterize_reader_nodule(
            annotation,
            source_image=image,
            ordered_sop_instance_uids=("SOP-0", "SOP-1", "SOP-2"),
        )


def test_reader_contour_rejects_degenerate_polygon() -> None:
    image = _source_image()
    annotation = ReaderNoduleAnnotation(
        reader_slot=1,
        reader_annotation_id="NODULE-2",
        characteristics=None,
        rois=(
            ROIReference(
                sop_instance_uid="SOP-1",
                z_position_mm=0.0,
                inclusion=True,
                edge_points=((5, 5), (6, 6)),
            ),
        ),
    )

    with pytest.raises(LIDCContourError, match="at least three"):
        rasterize_reader_nodule(
            annotation,
            source_image=image,
            ordered_sop_instance_uids=("SOP-0", "SOP-1", "SOP-2"),
        )
