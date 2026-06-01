from src.dataset.yolo_converter import YOLOConverter
from src.dataset.class_map import CLASS_MAP
from src.dataset.yolo_yaml import YoloYamlGenerator
from src.training.trainer import YOLOTrainer

from src.utils.paths import REPORT_DIR


def main():

    converter = YOLOConverter(CLASS_MAP)

    # Convert train
    converter.convert_csv(
        csv_path=str(REPORT_DIR / "train_objects.csv"),
        output_label_dir="data/bdd100k/labels/100k/train"
    )

    # Convert val
    converter.convert_csv(
        csv_path=str(REPORT_DIR / "val_objects.csv"),
        output_label_dir="data/bdd100k/labels/100k/val"
    )

    # Generate dataset.yaml
    yaml_gen = YoloYamlGenerator(
        dataset_root="data/bdd100k",
        class_names=list(CLASS_MAP.keys())
    )

    yaml_path = yaml_gen.generate()

    # Train
    trainer = YOLOTrainer("yolov8n.pt")

    trainer.train(
        data_yaml=str(yaml_path),
        epochs=50,
        imgsz=640,
        batch=16
    )


if __name__ == "__main__":
    main()