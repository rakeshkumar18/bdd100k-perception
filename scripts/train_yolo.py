"""
Train a YOLO model on the BDD100K dataset.
"""

import argparse
import time

from ultralytics import settings

from src.training.seed import set_seed
from src.training.trainer import YOLOTrainer
from src.tracking.mlflow_logger import MLflowLogger
from src.tracking.mlflow_tracker import YOLOMLflowTracker
from src.utils.config_loader import load_training_config

settings.update({"mlflow": False})


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed CLI arguments.
    """

    parser = argparse.ArgumentParser(
        description="Train YOLO on BDD100K."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/training.yaml",
        help="Path to training configuration file.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        help="Override number of training epochs.",
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
        "--fraction",
        type=float,
        help="Override training data fraction.",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="YOLO model checkpoint.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previous training run.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Execute YOLO training workflow.
    """

    args = parse_args()

    config = load_training_config()

    # ==========================================================
    # CLI overrides
    # ==========================================================

    if args.epochs is not None:
        config.epochs = args.epochs

    if args.batch is not None:
        config.batch = args.batch

    if args.imgsz is not None:
        config.imgsz = args.imgsz

    if args.fraction is not None:
        config.fraction = args.fraction

    if args.model is not None:
        config.model_name = args.model

    config.resume = args.resume

    # ==========================================================
    # Reproducibility
    # ==========================================================

    set_seed(
        config.seed
    )

    # ==========================================================
    # Training setup
    # ==========================================================

    trainer = YOLOTrainer(
        config=config
    )

    logger = MLflowLogger(
        experiment_name=config.experiment_name,
    )

    tracker = YOLOMLflowTracker()

    print("\n========== TRAINING CONFIG ==========")
    print(f"Model      : {config.model_name}")
    print(f"Epochs     : {config.epochs}")
    print(f"Batch      : {config.batch}")
    print(f"Image Size : {config.imgsz}")
    print(f"Fraction   : {config.fraction}")
    print(f"Device     : {config.device}")
    print(f"Workers    : {config.workers}")
    print(f"Cache      : {config.cache}")
    print(f"Run Name   : {config.run_name}")
    print("=====================================\n")

    with logger.start_run(
        run_name=config.run_name,
    ):
        try:
            start_time = time.time()

            results = trainer.train()

            training_time = (
                time.time() - start_time
            )

            tracker.log_run(
                config=config,
                results=results,
                training_time=training_time,
            )

            logger.set_tag(
                "training_status",
                "completed",
            )

            print(
                f"\nTraining completed in "
                f"{training_time:.2f} seconds."
            )

        except Exception as exc:
            logger.set_tags(
                {
                    "training_status": "failed",
                    "error": str(exc),
                }
            )

            raise


if __name__ == "__main__":
    main()