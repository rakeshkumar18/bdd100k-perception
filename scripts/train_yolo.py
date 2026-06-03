from src.training.trainer import YOLOTrainer


def main():

    trainer = YOLOTrainer(
        model_name="yolov8n.pt"
    )

    trainer.train(
        data_yaml="configs/yolo_dataset.yaml",
        epochs=1,
        imgsz=640,
        batch=8,
    )


if __name__ == "__main__":
    main()
