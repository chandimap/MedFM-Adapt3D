"""Explicit parser for reader-specific LIDC-IDRI XML annotations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

from medfm_adapt3d.data.lidc_schema import (
    NoduleCharacteristics,
    ReaderNoduleAnnotation,
    ROIReference,
)


@dataclass(frozen=True, slots=True)
class LIDCXMLScan:
    """Annotation payload tied to one DICOM study/series."""

    study_instance_uid: str
    series_instance_uid: str
    annotations: tuple[ReaderNoduleAnnotation, ...]
    source_xml: str


def _local_name(tag: str) -> str:
    """Stripping any XML namespace from an element tag."""
    return tag.rsplit("}", maxsplit=1)[-1]


def _children_by_name(element: ET.Element, name: str) -> list[ET.Element]:
    return [
        child
        for child in element
        if _local_name(child.tag).casefold() == name.casefold()
    ]


def _descendants_by_name(element: ET.Element, name: str) -> list[ET.Element]:
    return [
        child
        for child in element.iter()
        if _local_name(child.tag).casefold() == name.casefold()
    ]


def _first_descendant_text(
    element: ET.Element,
    names: tuple[str, ...],
) -> str | None:
    wanted = {name.casefold() for name in names}

    for child in element.iter():
        if _local_name(child.tag).casefold() not in wanted:
            continue

        text = (child.text or "").strip()
        if text:
            return text

    return None


def _optional_int(element: ET.Element, name: str) -> int | None:
    text = _first_descendant_text(element, (name,))
    if text is None:
        return None

    try:
        return int(text)
    except ValueError as error:
        raise ValueError(f"Invalid integer for LIDC field {name!r}: {text!r}") from error


def _required_float(element: ET.Element, name: str) -> float:
    text = _first_descendant_text(element, (name,))
    if text is None:
        raise ValueError(f"Missing required LIDC field {name!r}.")

    try:
        return float(text)
    except ValueError as error:
        raise ValueError(f"Invalid float for LIDC field {name!r}: {text!r}") from error


def _parse_characteristics(element: ET.Element) -> NoduleCharacteristics | None:
    matches = _children_by_name(element, "characteristics")

    if not matches:
        return None

    characteristics = matches[0]

    values = {
        "subtlety": _optional_int(characteristics, "subtlety"),
        "internal_structure": _optional_int(characteristics, "internalStructure"),
        "calcification": _optional_int(characteristics, "calcification"),
        "sphericity": _optional_int(characteristics, "sphericity"),
        "margin": _optional_int(characteristics, "margin"),
        "lobulation": _optional_int(characteristics, "lobulation"),
        "spiculation": _optional_int(characteristics, "spiculation"),
        "texture": _optional_int(characteristics, "texture"),
        "malignancy": _optional_int(characteristics, "malignancy"),
    }

    # Some XML entries structurally contain a characteristics node but do not
    # contain a valid semantic assessment. Do not manufacture labels from them.
    if all(value is None or value == 0 for value in values.values()):
        return None

    return NoduleCharacteristics(**values)


def _parse_roi(element: ET.Element) -> ROIReference:
    sop_uid = _first_descendant_text(
        element,
        (
            "imageSOP_UID",
            "imageSOPUID",
        ),
    )

    if sop_uid is None:
        raise ValueError("LIDC ROI is missing image SOP Instance UID.")

    z_position = _required_float(element, "imageZposition")

    inclusion_text = _first_descendant_text(element, ("inclusion",))
    inclusion = inclusion_text is None or inclusion_text.casefold() == "true"

    points: list[tuple[int, int]] = []

    for edge_map in _children_by_name(element, "edgeMap"):
        x_text = _first_descendant_text(edge_map, ("xCoord",))
        y_text = _first_descendant_text(edge_map, ("yCoord",))

        if x_text is None or y_text is None:
            raise ValueError("LIDC edgeMap is missing xCoord or yCoord.")

        points.append((int(x_text), int(y_text)))

    return ROIReference(
        sop_instance_uid=sop_uid,
        z_position_mm=z_position,
        inclusion=inclusion,
        edge_points=tuple(points),
    )


def _parse_reader_annotation(
    element: ET.Element,
    *,
    reader_slot: int,
) -> ReaderNoduleAnnotation:
    annotation_id = _first_descendant_text(element, ("noduleID",))

    if annotation_id is None:
        raise ValueError("LIDC unblindedReadNodule is missing noduleID.")

    rois = tuple(_parse_roi(roi) for roi in _children_by_name(element, "roi"))

    if not rois:
        raise ValueError(
            f"LIDC annotation {annotation_id!r} contains no ROI references."
        )

    return ReaderNoduleAnnotation(
        reader_slot=reader_slot,
        reader_annotation_id=annotation_id,
        characteristics=_parse_characteristics(element),
        rois=rois,
    )


def parse_lidc_xml(path: str | Path) -> LIDCXMLScan:
    """Parsing one final LIDC XML document without inventing reader consensus."""
    xml_path = Path(path)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    study_uid = _first_descendant_text(
        root,
        (
            "StudyInstanceUID",
            "StudyInstanceUid",
        ),
    )
    series_uid = _first_descendant_text(
        root,
        (
            "SeriesInstanceUID",
            "SeriesInstanceUid",
        ),
    )

    if study_uid is None:
        raise ValueError(f"{xml_path}: missing StudyInstanceUID.")

    if series_uid is None:
        raise ValueError(f"{xml_path}: missing SeriesInstanceUID.")

    annotations: list[ReaderNoduleAnnotation] = []

    sessions = _descendants_by_name(root, "readingSession")

    for reader_slot, session in enumerate(sessions, start=1):
        for nodule in _children_by_name(session, "unblindedReadNodule"):
            annotations.append(
                _parse_reader_annotation(
                    nodule,
                    reader_slot=reader_slot,
                )
            )

    return LIDCXMLScan(
        study_instance_uid=study_uid,
        series_instance_uid=series_uid,
        annotations=tuple(annotations),
        source_xml=str(xml_path),
    )