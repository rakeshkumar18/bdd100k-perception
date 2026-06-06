"""Convert processed BDD100K object CSV files into YOLO label files."""

from pathlib import Path

import pandas as pd

from src.dataset.yolo_converter import YOLOConverter
from src.utils.paths import REPORT_DIR

CLASS_MAP = {
    "car": 0,
    "truck": 1,
    "bus": 2,
    "person": 3,
    "rider": 4,
    "bike": 5,
    "motor": 6,
    "traffic light": 7,
    "traffic sign": 8,
    "train": 9,
}


def convert_split(csv_name: str, output_dir: Path) -> None:
    """Convert one processed split CSV into YOLO-format label files.

    Args:
        csv_name: Name of the object-level CSV file stored in the report directory.
        output_dir: Directory where YOLO label files will be written.
    """
    df = pd.read_csv(REPORT_DIR / csv_name)

    converter = YOLOConverter(class_map=CLASS_MAP)
    converter.convert_df(df=df, output_dir=output_dir)


def main() -> None:
    """Convert train and validation object annotations into YOLO labels."""
    convert_split("train_objects.csv", REPORT_DIR / "yolo_labels/train")
    convert_split("val_objects.csv", REPORT_DIR / "yolo_labels/val")


if __name__ == "__main__":
    main()
