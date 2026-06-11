import platform
from pathlib import Path

import mlflow
import torch


class YOLOMLflowTracker:

    def log_run(
        self,
        config,
        results,
        training_time,
    ):

        results_dict = results.results_dict

        run_dir = Path(results.save_dir)
        mlflow.set_tag(
            "run_dir",
            str(run_dir)
        )

        best_model_path = run_dir / "weights" / "best.pt"

        mlflow.log_params(
            {
                "model": config.model_name,
                "epochs": config.epochs,
                "imgsz": config.imgsz,
                "batch": config.batch,
                "device": config.device,
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
            }
        )

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

        mlflow.set_tag(
            "best_model_path",
            str(best_model_path),
        )

        mlflow.log_artifacts(str(run_dir))