"""Run batch YOLO inference on BDD100K validation images and save predictions."""

from pathlib import Path

from src.inference.batch_predict import get_images
from src.inference.predictor import YOLOPredictor
from src.inference.visualize import save_prediction_image


def main() -> None:
    """Run batch inference on a subset of validation images.

    The script loads a trained YOLO model, collects validation images from the
    BDD100K dataset directory, and saves annotated prediction images for the
    first 20 files.
    """
    predictor = YOLOPredictor(
        model_path="runs/detect/outputs/training/yolov8n_bdd100k/weights/best.pt",
    )

    images = get_images("../datasets/bdd100k/images/100k/val")
    print(f"Found {len(images)} images")

    output_dir = Path("outputs/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in images[:20]:
        print(f"Processing {image_path.name}")
        results = predictor.predict(str(image_path))
        save_prediction_image(
            results[0],
            str(output_dir / f"{image_path.stem}_pred.jpg"),
        )


if __name__ == "__main__":
    main()
