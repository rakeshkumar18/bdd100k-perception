"""MLflow run table helpers."""

from typing import Any

import pandas as pd
import streamlit as st


def get_metric_column(
    runs: pd.DataFrame,
    metric_name: str,
) -> pd.Series | None:
    """Return metric column."""

    new_col = (
        f"metrics.{metric_name}"
    )

    old_col = (
        f"metrics.metrics/{metric_name}"
    )

    if new_col in runs.columns:

        values = runs[new_col]

        if old_col in runs.columns:

            values = values.fillna(
                runs[old_col]
            )

        return values

    if old_col in runs.columns:
        return runs[old_col]

    return None


def get_best_run(
    runs: pd.DataFrame,
) -> pd.Series:
    """Return best historical run."""

    runs = runs.copy()

    metric_col = get_metric_column(
        runs,
        "mAP50-95B",
    )

    if metric_col is None:

        raise ValueError(
            "mAP50-95 metric column not found."
        )

    runs["mAP50_95"] = metric_col

    runs = runs.dropna(
        subset=["mAP50_95"]
    )

    if runs.empty:

        raise ValueError(
            "No completed runs found."
        )

    return runs.loc[
        runs["mAP50_95"].idxmax()
    ]


def get_latest_valid_run(
    runs: pd.DataFrame,
) -> pd.Series:
    """Return newest completed run."""

    metric_candidates = [
        "metrics.mAP50B",
        "metrics.metrics/mAP50B",
    ]

    valid_runs = pd.DataFrame()

    for col in metric_candidates:

        if col not in runs.columns:
            continue

        valid_runs = runs[
            runs[col].notna()
        ]

        break

    if valid_runs.empty:
        return runs.iloc[0]

    for candidate in [
        "start_time",
        "start_time_ms",
        "creation_time",
    ]:

        if candidate not in valid_runs.columns:
            continue

        valid_runs = valid_runs.sort_values(
            by=candidate,
            ascending=False,
        )

        break

    return valid_runs.iloc[0]


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


def show_run_comparison(
    runs: pd.DataFrame,
) -> pd.DataFrame:
    """Display run comparison table."""

    comparison_df = runs.copy()

    comparison_df["mAP50"] = (
        get_metric_column(
            comparison_df,
            "mAP50B",
        )
    )

    comparison_df["mAP50-95"] = (
        get_metric_column(
            comparison_df,
            "mAP50-95B",
        )
    )

    comparison_df["Precision"] = (
        get_metric_column(
            comparison_df,
            "precisionB",
        )
    )

    comparison_df["Recall"] = (
        get_metric_column(
            comparison_df,
            "recallB",
        )
    )

    comparison_df = comparison_df[
        comparison_df["status"]
        == "FINISHED"
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
        comparison_df[
            display_cols
        ].sort_values(
            by="mAP50-95",
            ascending=False,
        ),
        width="stretch",
    )

    return comparison_df