"""
YOLO inference utilities.
"""

from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from src.model_registry.model_registry import ModelRegistry


class YOLOPredictor:
    """Wrapper around Ultralytics YOLO inference."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        experiment_name: str = "BDD100K_YOLO",
    ) -> None:
        """
        Initialize predictor.

        Args:
            model_path:
                Optional path to a model. If not provided,
                the best model from the registry is loaded.

            experiment_name:
                MLflow experiment name.
        """

        if model_path is None:
            registry = ModelRegistry(
                experiment_name=experiment_name,
            )
            model_path = registry.get_best_model_path()

        self.model_path = Path(model_path)
        self.model = YOLO(str(self.model_path))

    def predict(
        self,
        image_path: str | Path,
        conf: float = 0.25,
    ) -> Results:
        """
        Run inference on a single image.

        Args:
            image_path:
                Input image path.

            conf:
                Confidence threshold.

        Returns:
            Ultralytics prediction result.
        """

        results = self.model.predict(
            source=str(image_path),
            conf=conf,
            verbose=False,
        )

        return results[0]

    def predict_frame(
        self,
        frame: np.ndarray,
        conf: float = 0.25,
    ) -> Results:
        """
        Run inference on an OpenCV frame.

        Args:
            frame:
                BGR image frame.

            conf:
                Confidence threshold.

        Returns:
            Ultralytics prediction result.
        """

        results = self.model.predict(
            source=frame,
            conf=conf,
            verbose=False,
        )

        return results[0]

    def track_frame(
        self,
        frame: np.ndarray,
        conf: float = 0.25,
    ) -> Results:
        """
        Run object tracking on an OpenCV frame.

        Args:
            frame:
                BGR image frame.

            conf:
                Confidence threshold.

        Returns:
            Ultralytics tracking result.
        """

        results = self.model.track(
            source=frame,
            conf=conf,
            persist=True,
            verbose=False,
        )

        return results[0]
