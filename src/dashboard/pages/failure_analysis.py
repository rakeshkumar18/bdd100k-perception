"""Failure analysis dashboard page."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.dashboard.mlflow_client import MLflowClient
from src.dashboard.pages.training import (
    get_best_run,
    get_metric,
)


# ==========================================================
# HELPERS
# ==========================================================


def show_image_if_exists(
    image_path: Path,
    caption: str,
) -> None:
    """Display image if available."""

    if image_path.exists():

        st.image(
            str(image_path),
            caption=caption,
            width="stretch",
        )

    else:

        st.warning(
            f"Missing file: {image_path.name}"
        )


def show_metric_summary(
    run: pd.Series,
) -> None:
    """Display top-level metrics."""

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "mAP50",
        f"{get_metric(run, 'mAP50B'):.3f}",
    )

    c2.metric(
        "mAP50_95B",
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


def load_results_csv(
    run_dir: Path,
) -> pd.DataFrame | None:
    """Load YOLO results.csv."""

    results_csv = run_dir / "results.csv"

    if not results_csv.exists():
        return None

    return pd.read_csv(
        results_csv
    )


# ==========================================================
# TRAINING FAILURE SIGNALS
# ==========================================================


def show_loss_analysis(
    run_dir: Path,
) -> None:
    """Analyze loss behaviour."""

    st.subheader(
        "Loss Analysis"
    )

    results = load_results_csv(
        run_dir
    )

    if results is None:
        st.warning(
            "results.csv not found."
        )
        return

    latest_epoch = results.iloc[-1]

    cols = st.columns(3)

    if "train/box_loss" in latest_epoch:
        cols[0].metric(
            "Final Box Loss",
            f"{latest_epoch['train/box_loss']:.4f}",
        )

    if "train/cls_loss" in latest_epoch:
        cols[1].metric(
            "Final Class Loss",
            f"{latest_epoch['train/cls_loss']:.4f}",
        )

    if "train/dfl_loss" in latest_epoch:
        cols[2].metric(
            "Final DFL Loss",
            f"{latest_epoch['train/dfl_loss']:.4f}",
        )

    st.info(
        """
        High box loss:
        localization issue

        High cls loss:
        class confusion issue

        High DFL loss:
        poor bounding box quality
        """
    )


# ==========================================================
# CONFUSION MATRIX REVIEW
# ==========================================================


def show_confusion_review(
    run_dir: Path,
) -> None:
    """Review confusion matrix."""

    st.subheader(
        "Class Confusion Analysis"
    )

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

    st.info(
        """
        Look for:

        • car ↔ truck confusion

        • bike ↔ motor confusion

        • rider ↔ person confusion

        • traffic sign ↔ traffic light confusion

        Strong off-diagonal cells indicate
        systematic classification errors.
        """
    )


# ==========================================================
# CLASS IMBALANCE REVIEW
# ==========================================================


def show_dataset_imbalance() -> None:
    """Display class imbalance report."""

    st.subheader(
        "Dataset Imbalance"
    )

    csv_path = (
        Path("outputs/reports")
        / "train_class_distribution.csv"
    )

    if not csv_path.exists():

        st.warning(
            "Class distribution report missing."
        )

        return

    df = pd.read_csv(
        csv_path
    )

    st.dataframe(
        df,
        width="stretch",
    )

    st.info(
        """
        Classes with low sample count
        often produce:

        • lower recall

        • unstable AP

        • more false negatives
        """
    )


# ==========================================================
# FAILURE CHECKLIST
# ==========================================================


def show_failure_checklist() -> None:
    """Manual review checklist."""

    st.subheader(
        "Recommended Failure Review"
    )

    st.markdown(
        """
### Small Objects
- Traffic lights
- Traffic signs
- Distant riders

### Night Scenes
- Dark highways
- Low illumination

### Weather
- Rain
- Fog

### Heavy Occlusion
- Crowded traffic
- Pedestrians behind vehicles

### Rare Classes
- Train
- Bus
- Rider

### Localization Errors
- Truncated vehicles
- Partial objects
- Border objects
"""
    )


# ==========================================================
# PAGE
# ==========================================================


def render() -> None:
    """Render failure analysis page."""

    st.title(
        "Failure Analysis"
    )

    client = MLflowClient()

    runs = client.get_runs()

    if runs.empty:

        st.warning(
            "No runs available."
        )

        return

    best_run = get_best_run(
        runs
    )

    st.subheader(
        "Best Historical Model"
    )

    show_metric_summary(
        best_run
    )

    st.divider()

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

    show_loss_analysis(
        run_dir
    )

    st.divider()

    show_confusion_review(
        run_dir
    )

    st.divider()

    show_dataset_imbalance()

    st.divider()

    show_failure_checklist()