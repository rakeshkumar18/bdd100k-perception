"""
MLflow utilities used by the Streamlit dashboard.
"""

import mlflow
import pandas as pd

from src.utils.paths import MLFLOW_DIR, MLFLOW_TRACKING_URI


class MLflowClient:
    """
    Read MLflow experiment information for dashboard visualizations.
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
    ) -> None:
        """
        Initialize MLflow client.

        Args:
            tracking_uri:
                Optional MLflow tracking URI.
        """

        if tracking_uri is None:
            tracking_uri = MLFLOW_TRACKING_URI

        mlflow.set_tracking_uri(tracking_uri)

    def get_runs(
        self,
        experiment_name: str = "BDD100K_YOLO",
    ) -> pd.DataFrame:
        """
        Retrieve all runs for an experiment.

        Args:
            experiment_name:
                MLflow experiment name.

        Returns:
            DataFrame containing all runs.
        """

        experiment = mlflow.get_experiment_by_name(experiment_name)

        if experiment is None:
            return pd.DataFrame()

        return mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    def get_best_run(
        self,
        experiment_name: str = "BDD100K_YOLO",
    ):
        """
        Return best run based on mAP50.

        Args:
            experiment_name:
                MLflow experiment name.

        Returns:
            Best MLflow run.
        """

        runs = self.get_runs(experiment_name)

        if runs.empty:
            return None

        runs = runs.dropna(subset=["metrics.mAP50B"])

        if runs.empty:
            return None

        return runs.loc[runs["metrics.mAP50B"].idxmax()]

    def get_latest_run(
        self,
        experiment_name: str = "BDD100K_YOLO",
    ):
        """
        Return latest MLflow run.

        Args:
            experiment_name:
                MLflow experiment name.

        Returns:
            Latest run.
        """

        runs = self.get_runs(experiment_name)

        if runs.empty:
            return None

        return runs.sort_values(
            "start_time",
            ascending=False,
        ).iloc[0]
