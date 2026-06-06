"""Train a YOLO model on the prepared BDD100K detection dataset."""

from ultralytics import settings

from src.training.trainer import YOLOTrainer

settings.update({"mlflow": False})


def main() -> None:
    """Initialize the YOLO trainer and start model training."""
    trainer = YOLOTrainer(model_name="yolov8n.pt")

    trainer.train(
        data_yaml="configs/yolo_dataset.yaml",
        epochs=2,
        imgsz=640,
        batch=8,
    )


if __name__ == "__main__":
    main()
