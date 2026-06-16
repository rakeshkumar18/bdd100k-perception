"""Model evaluation dashboard page."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.dashboard.mlflow_client import MLflowClient
from src.dashboard.pages.training import (
    get_best_run,
    get_metric,
)

def show_image_if_exists(
    image_path: Path,
    caption: str,
) -> None:
    """Display image if present."""

    if image_path.exists():

        st.image(
            str(image_path),
            caption=caption,
            width="stretch",
        )

    else:

        st.warning(
            f"Missing artifact: {image_path.name}"
        )

def show_metrics(
    run: pd.Series,
) -> None:
    """Display evaluation metrics."""

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "mAP50",
        f"{get_metric(run, 'mAP50B'):.3f}",
    )

    c2.metric(
        "mAP50-95",
        f"{get_metric(run, 'mAP50-95B'):.3f}",
    )

    c3.metric(
        "Precision",
        f"{get_metric(run, 'precisionB'):.3f}",
    )

    c4.metric(
        "Recall",
        f"{get_metric(run, 'recallB'):.3f}",
    )

def show_metrics(
    run: pd.Series,
) -> None:
    """Display evaluation metrics."""

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "mAP50",
        f"{get_metric(run, 'mAP50B'):.3f}",
    )

    c2.metric(
        "mAP50_95",
        f"{get_metric(run, 'mAP50_95B'):.3f}",
    )

    c3.metric(
        "Precision",
        f"{get_metric(run, 'precisionB'):.3f}",
    )

    c4.metric(
        "Recall",
        f"{get_metric(run, 'recallB'):.3f}",
    )

def show_model_metadata(
    run: pd.Series,
) -> None:
    """Display selected run metadata."""

    st.subheader(
        "Model Metadata"
    )

    metadata = {}

    for col in [
        "run_id",
        "artifact_uri",
        "tags.best_model_path",
        "tags.run_dir",
        "start_time",
        "status",
    ]:

        if col in run.index:
            metadata[col] = run[col]

    st.json(
        metadata
    )

def show_detection_curves(
    run_dir: Path,
) -> None:
    """Display PR and F1 curves."""

    col1, col2 = st.columns(2)

    with col1:

        show_image_if_exists(
            run_dir / "BoxPR_curve.png",
            "Precision Recall Curve",
        )

    with col2:

        show_image_if_exists(
            run_dir / "BoxF1_curve.png",
            "F1 Curve",
        )

    col1, col2 = st.columns(2)

    with col1:

        show_image_if_exists(
            run_dir / "BoxP_curve.png",
            "Precision Curve",
        )

    with col2:

        show_image_if_exists(
            run_dir / "BoxR_curve.png",
            "Recall Curve",
        )

def show_confusion_matrix(
    run_dir: Path,
) -> None:
    """Display confusion matrices."""

    col1, col2 = st.columns(2)

    with col1:

        show_image_if_exists(
            run_dir / "confusion_matrix.png",
            "Confusion Matrix",
        )

    with col2:

        show_image_if_exists(
            run_dir
            / "confusion_matrix_normalized.png",
            "Normalized Confusion Matrix",
        )


def render() -> None:
    """Render evaluation page."""

    st.title(
        "Model Evaluation"
    )

    client = MLflowClient()

    runs = client.get_runs()

    if runs.empty:

        st.warning(
            "No runs found."
        )

        return

    best_run = get_best_run(
        runs
    )

    st.subheader(
        "Best Historical Model"
    )

    show_metrics(
        best_run
    )

    st.divider()

    st.subheader(
        "Run Selection"
    )

    selected_run_id = st.selectbox(
        "Select Run",
        runs["run_id"].tolist(),
    )

    selected_run = runs[
        runs["run_id"]
        == selected_run_id
    ].iloc[0]

    run_dir = Path(
        selected_run[
            "tags.run_dir"
        ]
    )

    st.divider()

    st.subheader(
        "Evaluation Metrics"
    )

    show_metrics(
        selected_run
    )

    st.divider()

    st.subheader(
        "Confusion Matrix"
    )

    show_confusion_matrix(
        run_dir
    )

    st.divider()

    st.subheader(
        "Detection Curves"
    )

    show_detection_curves(
        run_dir
    )

    st.divider()

    show_model_metadata(
        selected_run
    )

