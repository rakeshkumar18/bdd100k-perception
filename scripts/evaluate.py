from pprint import pprint

from src.evaluation.evaluator import (
    YOLOEvaluator
)

from src.evaluation.metrics import (
    extract_metrics
)


def main():

    evaluator = YOLOEvaluator(
        model_path=
        "runs/detect/outputs/training/yolov8n_bdd100k/weights/best.pt"
    )

    results = evaluator.evaluate(
        "configs/yolo_dataset.yaml"
    )

    metrics = extract_metrics(
        results
    )

    pprint(metrics)


if __name__ == "__main__":
    main()