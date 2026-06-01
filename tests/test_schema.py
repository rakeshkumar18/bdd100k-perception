from src.ingestion.schema import BoundingBox


def test_bbox_area():

    bbox = BoundingBox(
        x1=100,
        y1=200,
        x2=300,
        y2=400
    )

    assert bbox.width == 200
    assert bbox.height == 200
    assert bbox.area == 40000