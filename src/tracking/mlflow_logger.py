"""Utilities for configuring and logging MLflow experiment runs."""

from pathlib import Path
from typing import Any

import mlflow
from mlflow import ActiveRun


class MLflowLogger:
    """Wrap common MLflow setup and logging operations for local experiments."""

    def __init__(self, experiment_name: str) -> None:
        """Configure the MLflow tracking backend and active experiment.

        Args:
            experiment_name: Name of the MLflow experiment to create or reuse.
        """
        tracking_db = Path("outputs/mlflow/mlflow.db")
        tracking_db.parent.mkdir(parents=True, exist_ok=True)

        mlflow.set_tracking_uri(f"sqlite:///{tracking_db.resolve()}")
        print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str) -> ActiveRun:
        """Start and return a new MLflow run.

        Args:
            run_name: Human-readable name for the run.

        Returns:
            The active MLflow run object.
        """
        return mlflow.start_run(run_name=run_name)

    @staticmethod
    def log_params(params: dict[str, Any]) -> None:
        """Log a dictionary of MLflow parameters.

        Args:
            params: Parameter names and values to record.
        """
        mlflow.log_params(params)

    @staticmethod
    def log_metrics(metrics: dict[str, float]) -> None:
        """Log a dictionary of MLflow metrics.

        Args:
            metrics: Metric names and numeric values to record.
        """
        mlflow.log_metrics(metrics)

    @staticmethod
    def log_artifacts(directory: str | Path) -> None:
        """Log all files from a local directory as MLflow artifacts.

        Args:
            directory: Directory containing artifacts to upload.
        """
        mlflow.log_artifacts(str(directory))
