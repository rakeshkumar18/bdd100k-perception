from src.ingestion.parser import BDDParser
from src.ingestion.dataframe_builder import (
    DataFrameBuilder
)

from src.utils.paths import TRAIN_LABELS


def test_dataframe_creation():

    parser = BDDParser()

    scenes = parser.load_directory(
        TRAIN_LABELS,
        max_files=5
    )

    builder = DataFrameBuilder()

    df = builder.build(scenes)

    assert len(df) > 0

    required_columns = [
        "image_name",
        "category",
        "width",
        "height",
        "area",
    ]

    for column in required_columns:

        assert column in df.columns