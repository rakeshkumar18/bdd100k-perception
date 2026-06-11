"""
YOLO training implementation.
"""

from ultralytics import YOLO

from src.training.config import TrainingConfig


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

    def train(self):
        """
        Execute YOLO training.

        Returns:
            Ultralytics training results.
        """

        return self.model.train(
            data=self.config.data_yaml,
            epochs=self.config.epochs,
            batch=self.config.batch,
            imgsz=self.config.imgsz,
            device=self.config.device,
            workers=self.config.workers,
            cache=self.config.cache,
            project=self.config.project,
            name=self.config.run_name,
            hsv_h=self.config.hsv_h,
            hsv_s=self.config.hsv_s,
            hsv_v=self.config.hsv_v,
            degrees=self.config.degrees,
            translate=self.config.translate,
            scale=self.config.scale,
            fraction=self.config.fraction,
            fliplr=self.config.fliplr,
            exist_ok=False,
            verbose=True,
        )