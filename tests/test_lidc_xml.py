from pathlib import Path

from medfm_adapt3d.data.lidc_xml import parse_lidc_xml


def test_parse_reader_specific_lidc_annotation(tmp_path: Path) -> None:
    xml = """<?xml version="1.0"?>
<LidcReadMessage xmlns="http://www.nih.gov">
  <ResponseHeader>
    <StudyInstanceUID>1.2.3</StudyInstanceUID>
    <SeriesInstanceUid>4.5.6</SeriesInstanceUid>
  </ResponseHeader>

  <readingSession>
    <unblindedReadNodule>
      <noduleID>N1</noduleID>

      <characteristics>
        <subtlety>4</subtlety>
        <internalStructure>1</internalStructure>
        <calcification>6</calcification>
        <sphericity>3</sphericity>
        <margin>4</margin>
        <lobulation>2</lobulation>
        <spiculation>2</spiculation>
        <texture>5</texture>
        <malignancy>3</malignancy>
      </characteristics>

      <roi>
        <imageZposition>-101.5</imageZposition>
        <imageSOP_UID>1.2.840.1</imageSOP_UID>
        <inclusion>TRUE</inclusion>

        <edgeMap>
          <xCoord>100</xCoord>
          <yCoord>200</yCoord>
        </edgeMap>

        <edgeMap>
          <xCoord>101</xCoord>
          <yCoord>201</yCoord>
        </edgeMap>
      </roi>
    </unblindedReadNodule>
  </readingSession>

  <readingSession>
    <unblindedReadNodule>
      <noduleID>N7</noduleID>

      <roi>
        <imageZposition>-100.0</imageZposition>
        <imageSOP_UID>1.2.840.2</imageSOP_UID>
        <inclusion>TRUE</inclusion>

        <edgeMap>
          <xCoord>120</xCoord>
          <yCoord>220</yCoord>
        </edgeMap>
      </roi>
    </unblindedReadNodule>
  </readingSession>
</LidcReadMessage>
"""

    path = tmp_path / "annotation.xml"
    path.write_text(xml, encoding="utf-8")

    parsed = parse_lidc_xml(path)

    assert parsed.study_instance_uid == "1.2.3"
    assert parsed.series_instance_uid == "4.5.6"

    assert len(parsed.annotations) == 2

    first = parsed.annotations[0]
    second = parsed.annotations[1]

    assert first.reader_slot == 1
    assert first.reader_annotation_id == "N1"
    assert first.is_large_nodule
    assert first.characteristics is not None
    assert first.characteristics.malignancy == 3
    assert first.characteristics.spiculation == 2

    assert first.rois[0].sop_instance_uid == "1.2.840.1"
    assert first.rois[0].edge_points == (
        (100, 200),
        (101, 201),
    )

    assert second.reader_slot == 2
    assert second.reader_annotation_id == "N7"
    assert not second.is_large_nodule