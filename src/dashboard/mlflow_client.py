# src/dashboard/mlflow_client.py

import mlflow
import pandas as pd


class MLflowClient:

    def __init__(self, tracking_uri="sqlite:///outputs/mlflow/mlflow.db"):
        mlflow.set_tracking_uri(tracking_uri)

    def get_runs(self, experiment_name="BDD100K_YOLO"):

        exp = mlflow.get_experiment_by_name(experiment_name)

        if exp is None:
            return pd.DataFrame()

        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])

        return runs

    def get_best_run(
        self,
        experiment_name="BDD100K_YOLO"
    ):

        runs = self.get_runs(
            experiment_name
        )

        runs = runs.dropna(
            subset=["metrics.mAP50B"]
        )

        return runs.loc[
            runs["metrics.mAP50B"].idxmax()
        ]