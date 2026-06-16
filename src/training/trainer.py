"""
YOLO training implementation.
"""

from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

from src.training.config import TrainingConfig
from src.utils.paths import TRAINING_RUNS_DIR


class YOLOTrainer:
    """
    Wrapper around Ultralytics YOLO training.
    """

    def __init__(
        self,
        config: TrainingConfig,
    ) -> None:
        """
        Initialize trainer.

        Args:
            config:
                Training configuration.
        """

        self.config = config

        self.model = YOLO(config.model_name)

        self.run_dir: Path | None = None

    def train(self):
        """
        Execute YOLO training.

        Returns:
            Ultralytics training results.
        """

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        run_name = (
            f"{self.config.run_name}_{timestamp}"
        )

        results = self.model.train(
            data=self.config.data_yaml,
            epochs=self.config.epochs,
            batch=self.config.batch,
            imgsz=self.config.imgsz,
            device=self.config.device,
            workers=self.config.workers,
            cache=self.config.cache,
            fraction=self.config.fraction,
            resume=self.config.resume,
            hsv_h=self.config.hsv_h,
            hsv_s=self.config.hsv_s,
            hsv_v=self.config.hsv_v,
            degrees=self.config.degrees,
            translate=self.config.translate,
            scale=self.config.scale,
            fliplr=self.config.fliplr,
            project=str(TRAINING_RUNS_DIR),
            name=run_name,
            exist_ok=False,
            verbose=True,
        )

        self.run_dir = Path(results.save_dir)

        return results