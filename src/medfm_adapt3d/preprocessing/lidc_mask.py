"""Reader-Preserving Conversion of LIDC XML Contours into Aligned 3D Masks."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk

from medfm_adapt3d.data.lidc_schema import ReaderNoduleAnnotation, ROIReference
from medfm_adapt3d.preprocessing.config import FloatTriplet
from medfm_adapt3d.preprocessing.geometry import validate_scalar_3d_image
from medfm_adapt3d.preprocessing.quality import (
    measure_mask_geometry,
    validate_binary_mask,
)


class LIDCContourError(ValueError):
    """Raised when an LIDC reader contour cannot be mapped without ambiguity."""


@dataclass(frozen=True, slots=True)
class RasterizedReaderNodule:
    """One reader-local annotation represented on its source DICOM grid."""

    mask: sitk.Image
    candidate_center_mm_lps: FloatTriplet
    reader_slot: int
    reader_annotation_id: str
    referenced_sop_instance_uids: tuple[str, ...]


def _draw_boundary(
    canvas: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
) -> None:
    """Rasterizing one integer contour segment with Bresenham's algorithm."""
    x0, y0 = start
    x1, y1 = end
    delta_x = abs(x1 - x0)
    step_x = 1 if x0 < x1 else -1
    delta_y = -abs(y1 - y0)
    step_y = 1 if y0 < y1 else -1
    error = delta_x + delta_y

    while True:
        canvas[y0, x0] = True
        if x0 == x1 and y0 == y1:
            break
        doubled_error = 2 * error
        if doubled_error >= delta_y:
            error += delta_y
            x0 += step_x
        if doubled_error <= delta_x:
            error += delta_x
            y0 += step_y


def _polygon_mask(
    points_xy: tuple[tuple[int, int], ...],
    *,
    rows: int,
    columns: int,
) -> np.ndarray:
    """Filling an integer LIDC polygon using even-odd interior and explicit edges."""
    points = tuple(dict.fromkeys(points_xy))
    if len(points) < 3:
        raise LIDCContourError("A nodule ROI contour requires at least three points.")

    for x_coordinate, y_coordinate in points:
        if not 0 <= x_coordinate < columns or not 0 <= y_coordinate < rows:
            raise LIDCContourError(
                "LIDC contour coordinate lies outside the linked DICOM image."
            )

    x_values = np.asarray([point[0] for point in points], dtype=np.float64)
    y_values = np.asarray([point[1] for point in points], dtype=np.float64)
    minimum_x = int(x_values.min())
    maximum_x = int(x_values.max())
    minimum_y = int(y_values.min())
    maximum_y = int(y_values.max())
    query_y, query_x = np.mgrid[
        minimum_y : maximum_y + 1,
        minimum_x : maximum_x + 1,
    ]
    inside = np.zeros(query_x.shape, dtype=bool)

    previous = len(points) - 1
    for current in range(len(points)):
        x_current = x_values[current]
        y_current = y_values[current]
        x_previous = x_values[previous]
        y_previous = y_values[previous]
        crosses_query_row = (y_current > query_y) != (y_previous > query_y)
        denominator = y_previous - y_current
        if denominator != 0:
            intersection_x = (
                (x_previous - x_current)
                * (query_y - y_current)
                / denominator
                + x_current
            )
            inside ^= crosses_query_row & (query_x < intersection_x)
        previous = current

    result = np.zeros((rows, columns), dtype=bool)
    result[
        minimum_y : maximum_y + 1,
        minimum_x : maximum_x + 1,
    ] = inside

    closed_points = (*points, points[0])
    for start, end in zip(closed_points[:-1], closed_points[1:], strict=True):
        _draw_boundary(result, start, end)

    return result


def _verify_roi_slice_position(
    roi: ROIReference,
    *,
    image: sitk.Image,
    slice_index: int,
    tolerance_mm: float,
) -> None:
    slice_origin = image.TransformIndexToPhysicalPoint((0, 0, slice_index))
    if not math.isclose(
        float(slice_origin[2]),
        roi.z_position_mm,
        rel_tol=0.0,
        abs_tol=tolerance_mm,
    ):
        raise LIDCContourError(
            "LIDC ROI z position does not match its SOP-linked DICOM slice: "
            f"{roi.z_position_mm:.6f} vs {slice_origin[2]:.6f} mm."
        )


def rasterize_reader_nodule(
    annotation: ReaderNoduleAnnotation,
    *,
    source_image: sitk.Image,
    ordered_sop_instance_uids: tuple[str, ...],
    z_tolerance_mm: float = 0.1,
) -> RasterizedReaderNodule:
    """Rasterizing one reader annotation without manufacturing consensus labels."""
    validate_scalar_3d_image(source_image, name="source CT")
    if not math.isfinite(z_tolerance_mm) or z_tolerance_mm < 0:
        raise ValueError("z_tolerance_mm must be finite and non-negative.")
    if len(ordered_sop_instance_uids) != source_image.GetSize()[2]:
        raise LIDCContourError(
            "SOP Instance UID order does not match the CT slice dimension."
        )
    if len(set(ordered_sop_instance_uids)) != len(ordered_sop_instance_uids):
        raise LIDCContourError("SOP Instance UID order contains duplicates.")
    if not annotation.rois:
        raise LIDCContourError("Reader nodule annotation contains no ROI contours.")

    sop_to_slice = {
        sop_uid: index for index, sop_uid in enumerate(ordered_sop_instance_uids)
    }
    inclusions: dict[int, list[np.ndarray]] = defaultdict(list)
    exclusions: dict[int, list[np.ndarray]] = defaultdict(list)
    rows = source_image.GetSize()[1]
    columns = source_image.GetSize()[0]

    for roi in annotation.rois:
        slice_index = sop_to_slice.get(roi.sop_instance_uid)
        if slice_index is None:
            raise LIDCContourError(
                "LIDC ROI references a SOP Instance UID absent from the source CT."
            )
        _verify_roi_slice_position(
            roi,
            image=source_image,
            slice_index=slice_index,
            tolerance_mm=z_tolerance_mm,
        )
        plane = _polygon_mask(
            roi.edge_points,
            rows=rows,
            columns=columns,
        )
        target = inclusions if roi.inclusion else exclusions
        target[slice_index].append(plane)

    volume = np.zeros(
        (
            source_image.GetSize()[2],
            rows,
            columns,
        ),
        dtype=np.uint8,
    )
    orphan_exclusion_slices = set(exclusions) - set(inclusions)
    if orphan_exclusion_slices:
        raise LIDCContourError(
            "Exclusion contour exists on a slice without an inclusion contour."
        )

    for slice_index, planes in inclusions.items():
        included = np.logical_or.reduce(planes)
        if exclusions[slice_index]:
            included &= ~np.logical_or.reduce(exclusions[slice_index])
        volume[slice_index] = included.astype(np.uint8)

    mask = sitk.GetImageFromArray(volume, isVector=False)
    mask.CopyInformation(source_image)
    try:
        validate_binary_mask(mask, require_non_empty=True)
    except ValueError as error:
        raise LIDCContourError(
            "Reader annotation produced an empty or invalid nodule mask."
        ) from error

    geometry = measure_mask_geometry(mask)
    return RasterizedReaderNodule(
        mask=mask,
        candidate_center_mm_lps=geometry.centroid_mm_lps,
        reader_slot=annotation.reader_slot,
        reader_annotation_id=annotation.reader_annotation_id,
        referenced_sop_instance_uids=tuple(
            dict.fromkeys(roi.sop_instance_uid for roi in annotation.rois)
        ),
    )
