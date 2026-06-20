"""
Model registry utilities.
"""

from pathlib import Path

import mlflow

from src.utils.paths import MLFLOW_DIR, TRAINING_RUNS_DIR, MLFLOW_TRACKING_URI


class ModelRegistry:
    """
    Resolve and manage trained models.
    """

    def __init__(
        self,
        experiment_name: str,
    ) -> None:
        """
        Initialize model registry.

        Args:
            experiment_name:
                MLflow experiment name.
        """

        self.experiment_name = experiment_name

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    def get_best_run(self):
        """
        Return the best MLflow run based on mAP50.

        Returns:
            Pandas Series representing the best run.
        """

        experiment = mlflow.get_experiment_by_name(self.experiment_name)

        if experiment is None:
            raise ValueError(f"Experiment '{self.experiment_name}' not found.")

        runs = mlflow.search_runs([experiment.experiment_id])

        runs = runs.dropna(subset=["metrics.mAP50B"])

        if runs.empty:
            raise ValueError("No runs with mAP50B found.")

        runs = runs.sort_values(
            "metrics.mAP50B",
            ascending=False,
        )

        return runs.iloc[0]

    def get_latest_run_dir(self) -> Path:
        """
        Return latest training run directory.

        Returns:
            Latest run directory.
        """

        run_dirs = [path for path in TRAINING_RUNS_DIR.iterdir() if path.is_dir()]

        if not run_dirs:
            raise FileNotFoundError("No training runs found.")

        return max(
            run_dirs,
            key=lambda path: path.stat().st_mtime,
        )

    def get_latest_model_path(self) -> Path:
        """
        Return latest trained model.

        Returns:
            Path to best.pt from latest run.
        """

        latest_run = self.get_latest_run_dir()

        model_path = latest_run / "weights" / "best.pt"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        return model_path
