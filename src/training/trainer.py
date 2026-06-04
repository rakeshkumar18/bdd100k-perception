from pathlib import Path

import mlflow
from ultralytics import YOLO

from src.tracking.mlflow_logger import MLflowLogger


class YOLOTrainer:

    def __init__(
        self,
        model_name="yolov8n.pt"
    ):
        self.model = YOLO(model_name)

    def train(
        self,
        data_yaml,
        epochs=50,
        imgsz=640,
        batch=8,
    ):

        logger = MLflowLogger(
            experiment_name="BDD100K_YOLO"
        )

        with logger.start_run(
            run_name="yolov8n_bdd100k"
        ):
            
            mlflow.log_params({
                "model": "yolov8n.pt",
                "epochs": epochs,
                "imgsz": imgsz,
                "batch": batch,
                "device": "mps",
            })

            results = self.model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device="mps",
                workers=4,
                cache=True,
                project="outputs/training",
                name="yolov8n_bdd100k",
                exist_ok=True,
                verbose=True,
            )

            run_dir = Path(results.save_dir)

            mlflow.log_artifacts(str(run_dir))

        return results