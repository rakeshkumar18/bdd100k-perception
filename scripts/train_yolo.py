"""
Train a YOLO model on BDD100K.
"""

import argparse
import time

from datetime import datetime
from ultralytics import settings

from src.training.seed import set_seed
from src.training.trainer import YOLOTrainer
from src.tracking.mlflow_logger import MLflowLogger
from src.tracking.mlflow_tracker import YOLOMLflowTracker
from src.utils.config_loader import load_training_config

settings.update({"mlflow": False})


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.

    Returns:
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(
        description="Train YOLO on BDD100K."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        help="Override training epochs.",
    )

    parser.add_argument(
        "--batch",
        type=int,
        help="Override batch size.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        help="Override image size.",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="YOLO model checkpoint.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Execute training workflow.
    """

    args = parse_args()

    config = load_training_config()

    if args.epochs:
        config.epochs = args.epochs

    if args.batch:
        config.batch = args.batch

    if args.imgsz:
        config.imgsz = args.imgsz

    if args.model:
        config.model_name = args.model

    set_seed(config.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    config.run_name = (
        f"{config.run_name}_{timestamp}"
    )

    trainer = YOLOTrainer(config)

    logger = MLflowLogger(
        experiment_name=config.experiment_name,
    )

    tracker = YOLOMLflowTracker()

    with logger.start_run(
        run_name=config.run_name,
    ):
        start_time = time.time()

        results = trainer.train()

        training_time = time.time() - start_time

        tracker.log_run(
            config=config,
            results=results,
            training_time=training_time,
        )


if __name__ == "__main__":
    main()