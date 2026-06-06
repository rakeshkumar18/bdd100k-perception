"""Run a single-image YOLO prediction using the trained BDD100K model."""

from src.inference.predictor import YOLOPredictor


def main() -> None:
    """Load the trained model and run inference on one sample image."""
    predictor = YOLOPredictor(
        model_path="runs/detect/outputs/training/yolov8n_bdd100k/weights/best.pt",
    )

    results = predictor.predict(image_path="sample.jpg")
    print(results)


if __name__ == "__main__":
    main()
