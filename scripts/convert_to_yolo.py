from src.dataset.yolo_converter import YOLOConverter
from src.utils.paths import REPORT_DIR
import pandas as pd


CLASS_MAP = {
    "car": 0,
    "truck": 1,
    "bus": 2,
    "person": 3,
    "rider": 4,
    "bike": 5,
    "motorcycle": 6,
    "traffic light": 7,
    "traffic sign": 8
}


def main():

    df = pd.read_csv(
        REPORT_DIR / "train_objects.csv"
    )

    converter = YOLOConverter(
        class_map=CLASS_MAP
    )

    converter.convert_df(
        df,
        output_dir=REPORT_DIR / "yolo_labels/train"
    )


if __name__ == "__main__":
    main()