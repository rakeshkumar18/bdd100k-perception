"""Visualization helpers for the BDD100K Streamlit dashboard."""

import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure


def metric_bar_chart(df: pd.DataFrame, metric: str) -> Figure:
    """Create a bar chart for a selected run-level metric.

    Args:
        df: DataFrame containing MLflow run data.
        metric: Column name of the metric to visualize.

    Returns:
        A Plotly bar chart showing the metric value for each run.
    """
    return px.bar(df, x="run_id", y=metric, title=metric)


def create_loss_chart(df: pd.DataFrame) -> Figure:
    """Create a bar chart summarizing available training and validation losses.

    Args:
        df: DataFrame containing run metrics.

    Returns:
        A Plotly bar chart of all available loss values.

    Raises:
        ValueError: If no expected loss columns are present in the DataFrame.
    """
    loss_cols = [
        "metrics.train/box_loss",
        "metrics.train/cls_loss",
        "metrics.train/dfl_loss",
        "metrics.val/box_loss",
        "metrics.val/cls_loss",
        "metrics.val/dfl_loss",
    ]
    available_loss_cols = [column for column in loss_cols if column in df.columns]

    if not available_loss_cols:
        raise ValueError("No training or validation loss columns found in DataFrame.")

    melted = df[available_loss_cols].melt(
        var_name="Loss",
        value_name="Value",
    )

    return px.bar(
        melted,
        x="Loss",
        y="Value",
        title="Training & Validation Losses",
    )
