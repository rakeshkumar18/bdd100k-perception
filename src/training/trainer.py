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

        self.config = config
        self.run_dir: Path | None = None

        print("TRAINER DEBUG resume:", self.config.resume)
        print("TRAINER DEBUG type:", type(self.config.resume))

        if self.config.resume is not None:
            checkpoint_path = Path(self.config.resume)

            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Resume checkpoint not found: {checkpoint_path}"
                )

            print(f"Resuming training from: {checkpoint_path}")

            self.model = YOLO(checkpoint_path)

        else:
            self.model = YOLO(self.config.model_name)

    def train(self):
        """
        Execute YOLO training.

        Returns:
            Ultralytics training results.
        """

        if self.config.resume is not None:
            results = self.model.train(
                resume=True,
            )
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{self.config.run_name}_{timestamp}"

            results = self.model.train(
                data=self.config.data_yaml,
                epochs=self.config.epochs,
                batch=self.config.batch,
                imgsz=self.config.imgsz,
                device=self.config.device,
                workers=self.config.workers,
                cache=self.config.cache,
                fraction=self.config.fraction,
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
                save=True,
                save_period=1,
            )

        self.run_dir = Path(results.save_dir)

        return results