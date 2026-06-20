"""Inference helper functions."""

from pathlib import Path
from src.utils.paths import PROJECT_ROOT

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from src.dashboard.mlflow_client import MLflowClient
from src.dashboard.components.mlflow_helpers import (
    get_best_run,
)
from src.inference.predictor import (
    YOLOPredictor,
)


@st.cache_resource
def load_predictor() -> YOLOPredictor:
    """
    Load predictor using best historical model.
    """

    client = MLflowClient()

    runs = client.get_runs()

    if runs.empty:
        raise ValueError("No MLflow runs found.")

    best_run = get_best_run(runs)

    model_path = Path(best_run["tags.best_model_path"])

    # If MLflow stored an absolute path and it exists,
    # use it directly.
    if model_path.exists():
        resolved_model_path = model_path

    # Otherwise reconstruct relative to PROJECT_ROOT.
    else:
        outputs_idx = model_path.parts.index("outputs")

        relative_path = Path(*model_path.parts[outputs_idx:])

        resolved_model_path = PROJECT_ROOT / relative_path

    print("Resolved model path:", resolved_model_path)

    return YOLOPredictor(model_path=resolved_model_path)

    return YOLOPredictor(model_path=model_path)


def get_annotated_image(
    result,
) -> np.ndarray:
    """
    Convert YOLO result to image.
    """

    plotted = result.plot()

    plotted = cv2.cvtColor(
        plotted,
        cv2.COLOR_BGR2RGB,
    )

    return plotted


def extract_detections(
    result,
) -> list[dict]:
    """
    Extract detections into dataframe-ready format.
    """

    detections = []

    names = result.names

    for box in result.boxes:

        cls_id = int(box.cls.item())

        confidence = float(box.conf.item())

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()

        detections.append(
            {
                "class": names[cls_id],
                "confidence": round(
                    confidence,
                    4,
                ),
                "x1": round(x1, 1),
                "y1": round(y1, 1),
                "x2": round(x2, 1),
                "y2": round(y2, 1),
            }
        )

    return detections


def detections_to_dataframe(
    detections: list[dict],
) -> pd.DataFrame:
    """Convert detections to dataframe."""

    return pd.DataFrame(detections)
