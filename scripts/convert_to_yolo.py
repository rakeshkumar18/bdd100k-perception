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
    "motor": 6,
    "traffic light": 7,
    "traffic sign": 8,
    "train": 9,
}


def convert_split(csv_name, output_dir):

    df = pd.read_csv(REPORT_DIR / csv_name)

    converter = YOLOConverter(class_map=CLASS_MAP)

    converter.convert_df(df=df, output_dir=output_dir)


def main():

    convert_split("train_objects.csv", REPORT_DIR / "yolo_labels/train")

    convert_split("val_objects.csv", REPORT_DIR / "yolo_labels/val")


if __name__ == "__main__":
    main()
