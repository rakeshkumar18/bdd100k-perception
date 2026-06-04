import mlflow
from pathlib import Path


class MLflowLogger:

    def __init__(self, experiment_name: str):

        tracking_db = Path("outputs/mlflow/mlflow.db")

        tracking_db.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        mlflow.set_tracking_uri(
            f"sqlite:///{tracking_db.resolve()}"
        )

        print(
            "MLflow Tracking URI:",
            mlflow.get_tracking_uri()
        )

        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str):
        return mlflow.start_run(run_name=run_name)

    @staticmethod
    def log_params(params: dict):
        mlflow.log_params(params)

    @staticmethod
    def log_metrics(metrics: dict):
        mlflow.log_metrics(metrics)

    @staticmethod
    def log_artifacts(directory: str):
        mlflow.log_artifacts(directory)