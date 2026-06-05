from pathlib import Path

import mlflow


class ModelRegistry:

    def __init__(self):

        mlflow.set_tracking_uri(
            "sqlite:///outputs/mlflow/mlflow.db"
        )

    def get_best_run(self):

        exp = mlflow.get_experiment_by_name(
            "BDD100K_YOLO"
        )

        runs = mlflow.search_runs(
            [exp.experiment_id]
        )

        runs = runs.dropna(
            subset=["metrics.mAP50B"]
        )

        runs = runs.sort_values(
            "metrics.mAP50B",
            ascending=False
        )

        return runs.iloc[0]

    def get_best_model_path(self):

        return str(
            Path(
                "runs/detect/outputs/training/yolov8n_bdd100k/weights/best.pt"
            )
        )