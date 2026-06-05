from pathlib import Path

import mlflow
from ultralytics import YOLO

from src.tracking.mlflow_logger import MLflowLogger


class YOLOTrainer:

    def __init__(
        self,
        model_name="yolov8n.pt"
    ):
        self.model_name = model_name
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
                "model": self.model_name,
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

            results_dict = results.results_dict

            best_model_path = (
                run_dir /
                "weights" /
                "best.pt"
            )

            mlflow.set_tag(
                "best_model_path",
                str(best_model_path)
            )

            mlflow.log_metrics({
                "mAP50B": float(
                    results_dict.get(
                        "metrics/mAP50(B)",
                        0.0
                    )
                ),
                "mAP50-95B": float(
                    results_dict.get(
                        "metrics/mAP50-95(B)",
                        0.0
                    )
                ),
                "precisionB": float(
                    results_dict.get(
                        "metrics/precision(B)",
                        0.0
                    )
                ),
                "recallB": float(
                    results_dict.get(
                        "metrics/recall(B)",
                        0.0
                    )
                ),
            })

            mlflow.log_artifacts(str(run_dir))

        return results