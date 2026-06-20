from src.evaluation.evaluator import (
    YOLOEvaluator,
)

from src.evaluation.failure_analysis import (
    FailureAnalyzer,
)


def main():

    evaluator = YOLOEvaluator()

    results = evaluator.evaluate("configs/yolo_dataset.yaml")

    report = FailureAnalyzer.analyze(results)

    print("\nWorst Classes")
    print(report.worst_classes)

    print("\nLow Recall Classes")
    print(report.low_recall_classes)

    print("\nLow Precision Classes")
    print(report.low_precision_classes)


if __name__ == "__main__":
    main()
