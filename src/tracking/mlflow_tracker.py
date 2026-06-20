"""
MLflow tracking utilities for YOLO training.
"""

import platform
from pathlib import Path

import mlflow
import torch


class YOLOMLflowTracker:
    """
    Log YOLO training runs to MLflow.
    """

    def log_run(
        self,
        config,
        results,
        training_time: float,
    ) -> None:
        """
        Log training metadata, metrics, and artifacts.

        Args:
            config:
                Training configuration.
            results:
                Ultralytics training results.
            training_time:
                Total training duration in seconds.
        """

        results_dict = results.results_dict

        run_dir = Path(results.save_dir)

        best_model_path = run_dir / "weights" / "best.pt"

        last_model_path = run_dir / "weights" / "last.pt"

        # ==========================================================
        # Parameters
        # ==========================================================

        mlflow.log_params(
            {
                "model": config.model_name,
                "epochs": config.epochs,
                "batch": config.batch,
                "imgsz": config.imgsz,
                "fraction": config.fraction,
                "device": config.device,
                "workers": config.workers,
                "cache": config.cache,
                "seed": config.seed,
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
            }
        )

        # ==========================================================
        # Metrics
        # ==========================================================

        mlflow.log_metrics(
            {
                "mAP50B": float(
                    results_dict.get(
                        "metrics/mAP50(B)",
                        0.0,
                    )
                ),
                "mAP50_95B": float(
                    results_dict.get(
                        "metrics/mAP50-95(B)",
                        0.0,
                    )
                ),
                "precisionB": float(
                    results_dict.get(
                        "metrics/precision(B)",
                        0.0,
                    )
                ),
                "recallB": float(
                    results_dict.get(
                        "metrics/recall(B)",
                        0.0,
                    )
                ),
                "training_time_sec": training_time,
            }
        )

        # ==========================================================
        # Tags
        # ==========================================================

        mlflow.set_tags(
            {
                "run_dir": str(run_dir),
                "best_model_path": str(best_model_path),
                "last_model_path": str(last_model_path),
                "model_name": config.model_name,
                "run_name": run_dir.name,
            }
        )

        # ==========================================================
        # Artifacts
        # ==========================================================

        if run_dir.exists():
            try:
                mlflow.log_artifacts(str(run_dir))
            except Exception as exc:
                print(f"WARNING: Failed to log artifacts: {exc}")
