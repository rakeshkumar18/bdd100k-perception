"""Shared MLflow helper functions."""

from pathlib import Path
from typing import Any

import pandas as pd


def get_metric(
    row: pd.Series,
    metric_name: str,
) -> Any:
    """Return metric value from MLflow row."""

    candidates = [
        f"metrics.{metric_name}",
        f"metrics.metrics/{metric_name}",
    ]

    for col in candidates:
        if col in row.index:
            value = row[col]

            if pd.notna(value):
                return value

    return None


def get_metric_column(
    runs: pd.DataFrame,
    metric_name: str,
) -> pd.Series | None:
    """Return metric column from MLflow dataframe."""

    new_col = f"metrics.{metric_name}"
    old_col = f"metrics.metrics/{metric_name}"

    if new_col in runs.columns:
        values = runs[new_col]

        if old_col in runs.columns:
            values = values.fillna(runs[old_col])

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
        "mAP50_95B",
    )

    if metric_col is None:
        raise ValueError("mAP50_95B metric column not found.")

    runs["mAP50_95"] = metric_col

    runs = runs.dropna(subset=["mAP50_95"])

    if runs.empty:
        raise ValueError("No completed runs found.")

    return runs.loc[runs["mAP50_95"].idxmax()]


def get_latest_valid_run(
    runs: pd.DataFrame,
) -> pd.Series:
    """Return latest run containing metrics."""

    metric_candidates = [
        "metrics.mAP50B",
        "metrics.mAP50_95B",
        "metrics.metrics/mAP50B",
        "metrics.metrics/mAP50_95B",
    ]

    valid_runs = pd.DataFrame()

    for col in metric_candidates:
        if col in runs.columns:
            valid_runs = runs[runs[col].notna()]
            break

    if valid_runs.empty:
        return runs.iloc[0]

    for time_col in (
        "start_time",
        "start_time_ms",
        "creation_time",
    ):
        if time_col in valid_runs.columns:
            valid_runs = valid_runs.sort_values(
                by=time_col,
                ascending=False,
            )
            break

    return valid_runs.iloc[0]


def get_run_dir(
    run: pd.Series,
) -> Path | None:
    """
    Return training directory.

    Uses:
        tags.run_dir
    """

    run_dir = run.get("tags.run_dir")

    if pd.isna(run_dir):
        return None

    path = Path(run_dir)

    if path.exists():
        return path

    return None


def get_best_model_path(
    run: pd.Series,
) -> Path | None:
    """Return best.pt path."""

    model_path = run.get("tags.best_model_path")

    if pd.isna(model_path):
        return None

    path = Path(model_path)

    if path.exists():
        return path

    return None


def get_last_model_path(
    run: pd.Series,
) -> Path | None:
    """Return last.pt path."""

    model_path = run.get("tags.last_model_path")

    if pd.isna(model_path):
        return None

    path = Path(model_path)

    if path.exists():
        return path

    return None
