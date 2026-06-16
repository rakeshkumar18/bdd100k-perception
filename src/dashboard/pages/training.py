"""Training dashboard page."""

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.dashboard.mlflow_client import MLflowClient
from src.dashboard.visualizations import create_loss_chart, create_metric_chart
from src.dashboard.components.mlflow_helpers import (
    get_best_run,
    get_latest_valid_run,
    get_metric,
    get_metric_column,

)


# ==========================================================
# UI HELPERS
# ==========================================================


def display_run_metrics(
    run: pd.Series,
    title: str,
) -> None:
    """Display metric cards."""

    st.subheader(title)

    c1, c2, c3, c4 = st.columns(4)

    map50 = get_metric(run, "mAP50B")
    map5095 = get_metric(run, "mAP50_95B")
    precision = get_metric(run, "precisionB")
    recall = get_metric(run, "recallB")

    c1.metric(
        "mAP50",
        f"{map50:.3f}" if map50 is not None else "N/A",
    )

    c2.metric(
        "mAP50-95",
        f"{map5095:.3f}" if map5095 is not None else "N/A",
    )

    c3.metric(
        "Precision",
        f"{precision:.3f}" if precision is not None else "N/A",
    )

    c4.metric(
        "Recall",
        f"{recall:.3f}" if recall is not None else "N/A",
    )


def display_run_info(
    run: pd.Series,
    label: str,
) -> None:
    """Display run metadata."""

    st.markdown(
        f"**{label} Run ID:** "
        f"`{run.get('run_id', 'N/A')}`"
    )

    if "start_time" in run.index:
        st.markdown(
            f"**Start Time:** "
            f"{run['start_time']}"
        )

    if "status" in run.index:
        st.markdown(
            f"**Status:** "
            f"{run['status']}"
        )

    if "tags.run_dir" in run.index:
        st.markdown(
            f"**Run Directory:** "
            f"`{run['tags.run_dir']}`"
        )


# ==========================================================
# RUN COMPARISON
# ==========================================================


def show_run_comparison(
    runs: pd.DataFrame,
) -> pd.DataFrame:
    """Display run comparison table."""

    comparison_df = runs.copy()

    comparison_df["mAP50"] = get_metric_column(
        comparison_df,
        "mAP50B",
    )

    comparison_df["mAP50-95"] = get_metric_column(
        comparison_df,
        "mAP50_95B",
    )

    comparison_df["Precision"] = get_metric_column(
        comparison_df,
        "precisionB",
    )

    comparison_df["Recall"] = get_metric_column(
        comparison_df,
        "recallB",
    )

    comparison_df = comparison_df[
        comparison_df["status"] == "FINISHED"
    ]

    display_cols = [
        "run_id",
        "status",
        "mAP50",
        "mAP50-95",
        "Precision",
        "Recall",
    ]

    for col in [
        "start_time",
        "end_time",
    ]:
        if col in comparison_df.columns:
            display_cols.insert(
                1,
                col,
            )

    st.dataframe(
        comparison_df[display_cols].sort_values(
            by="mAP50-95",
            ascending=False,
        ),
        width="stretch",
    )

    return comparison_df


# ==========================================================
# TRAINING CURVES
# ==========================================================
def show_training_curves(
    run_dir: Path,
) -> None:

    results_csv = run_dir / "results.csv"

    if not results_csv.exists():
        st.warning(
            f"Missing results.csv: {results_csv}"
        )
        return

    try:
        results_df = pd.read_csv(
            run_dir / "results.csv"
        )
        st.plotly_chart(
            create_loss_chart(results_df),
            width="stretch",
        )

    except Exception as exc:
        st.error(
            f"Unable to load training curves: {exc}"
        )


# ==========================================================
# TRAINING CONFIGURATION
# ==========================================================


def show_training_configuration(
    run: pd.Series,
) -> None:
    """Display selected run parameters."""

    param_cols = sorted(
        [
            col
            for col in run.index
            if col.startswith("params.")
        ]
    )

    if not param_cols:
        st.info(
            "No parameters logged."
        )
        return

    config_df = pd.DataFrame(
        {
            "Parameter": [
                col.replace(
                    "params.",
                    "",
                )
                for col in param_cols
            ],
            "Value": [
                run[col]
                for col in param_cols
            ],
        }
    )

    st.dataframe(
        config_df,
        width="stretch",
    )


# ==========================================================
# ARTIFACTS
# ==========================================================


def show_artifacts(
    run_dir: Path,
) -> None:
    """Display training artifacts."""

    artifact_files = [
        ("results.png", "Training Metrics"),
        ("BoxPR_curve.png", "Precision Recall Curve"),
        ("BoxF1_curve.png", "F1 Curve"),
        ("BoxP_curve.png", "Precision Curve"),
        ("BoxR_curve.png", "Recall Curve"),
        ("confusion_matrix.png", "Confusion Matrix"),
        (
            "confusion_matrix_normalized.png",
            "Normalized Confusion Matrix",
        ),
    ]

    col1, col2 = st.columns(2)

    for idx, (
        filename,
        caption,
    ) in enumerate(
        artifact_files
    ):
        file_path = run_dir / filename

        if not file_path.exists():
            continue

        target_col = (
            col1
            if idx % 2 == 0
            else col2
        )

        with target_col:
            st.image(
                str(file_path),
                caption=caption,
                width="stretch",
            )


# ==========================================================
# PAGE
# ==========================================================


def render() -> None:
    """Render training dashboard."""

    st.title(
        "BDD100K Training Dashboard"
    )

    client = MLflowClient()

    runs = client.get_runs()

    if runs.empty:
        st.warning(
            "No runs found."
        )
        return

    latest_run = get_latest_valid_run(
        runs
    )

    best_run = get_best_run(
        runs
    )

    display_run_metrics(
        latest_run,
        "Latest Run",
    )

    display_run_info(
        latest_run,
        "Latest",
    )

    st.divider()

    display_run_metrics(
        best_run,
        "Best Historical Run",
    )

    display_run_info(
        best_run,
        "Best",
    )

    st.divider()

    st.subheader(
        "Run Comparison"
    )

    comparison_df = show_run_comparison(
        runs
    )

    st.divider()

    st.subheader(
        "Selected Run Analysis"
    )

    run_options = (
        comparison_df.sort_values(
            by="start_time",
            ascending=False,
        )["run_id"]
        .tolist()
    )

    selected_run_id = st.selectbox(
        "Select Run",
        run_options,
    )

    selected_run = runs[
        runs["run_id"]
        == selected_run_id
    ].iloc[0]

    display_run_metrics(
        selected_run,
        "Selected Run",
    )

    display_run_info(
        selected_run,
        "Selected",
    )

    if (
        "tags.run_dir"
        not in selected_run.index
    ):
        st.error(
            "tags.run_dir missing from MLflow run."
        )
        return

    run_dir = Path(
        selected_run["tags.run_dir"]
    )

    if not run_dir.exists():
        st.error(
            f"Run directory not found:\n{run_dir}"
        )
        return

    st.divider()

    st.subheader(
        "Training Configuration"
    )

    show_training_configuration(
        selected_run
    )

    st.divider()

    st.header(
        "Training Artifacts"
    )

    show_artifacts(
        run_dir
    )

    st.subheader(
    "Training Curves"
    )

    show_training_curves(
        run_dir
    )