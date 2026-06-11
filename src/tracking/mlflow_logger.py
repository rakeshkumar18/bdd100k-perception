"""
Utilities for configuring and interacting with MLflow.

This module provides a thin wrapper around MLflow for
experiment creation, run management, parameter logging,
metric logging, artifact logging, and tagging.
"""

from pathlib import Path
from typing import Any

import mlflow


class MLflowLogger:
    """Manage MLflow experiment configuration and logging."""

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str | None = None,
    ) -> None:
        """
        Initialize the MLflow tracking backend and experiment.

        Args:
            experiment_name:
                Name of the MLflow experiment.

            tracking_uri:
                Optional MLflow tracking URI. If not provided,
                a local SQLite database is created under
                outputs/mlflow/mlflow.db.
        """
        if tracking_uri is None:
            tracking_db = Path("outputs/mlflow/mlflow.db")

            tracking_db.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            tracking_uri = (
                f"sqlite:///{tracking_db.resolve()}"
            )

        mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)

    @staticmethod
    def start_run(
        run_name: str,
        nested: bool = False,
    ):
        """
        Start an MLflow run.

        Args:
            run_name:
                Human-readable run name.

            nested:
                Whether the run should be nested under
                an existing parent run.

        Returns:
            Active MLflow run.
        """
        return mlflow.start_run(
            run_name=run_name,
            nested=nested,
        )

    @staticmethod
    def log_params(
        params: dict[str, Any],
    ) -> None:
        """
        Log multiple parameters.

        Args:
            params:
                Dictionary of parameter names and values.
        """
        mlflow.log_params(params)

    @staticmethod
    def log_metrics(
        metrics: dict[str, float],
    ) -> None:
        """
        Log multiple metrics.

        Args:
            metrics:
                Dictionary of metric names and values.
        """
        mlflow.log_metrics(metrics)

    @staticmethod
    def log_artifacts(
        directory: str | Path,
    ) -> None:
        """
        Log all files from a directory as MLflow artifacts.

        Args:
            directory:
                Directory containing artifacts.
        """
        mlflow.log_artifacts(str(directory))

    @staticmethod
    def set_tag(
        key: str,
        value: str,
    ) -> None:
        """
        Set a single MLflow tag.

        Args:
            key:
                Tag name.

            value:
                Tag value.
        """
        mlflow.set_tag(key, value)

    @staticmethod
    def set_tags(
        tags: dict[str, str],
    ) -> None:
        """
        Set multiple MLflow tags.

        Args:
            tags:
                Dictionary of tag names and values.
        """
        mlflow.set_tags(tags)