"""Visualization helpers for the BDD100K Streamlit dashboard."""

import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure


def create_metric_chart(
    df: pd.DataFrame,
) -> Figure:

    metric_columns = [
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]

    available = [
        col
        for col in metric_columns
        if col in df.columns
    ]

    fig = px.line(
        df,
        x="epoch",
        y=available,
        title="Validation Metrics",
    )

    return fig

def create_loss_chart(
    df: pd.DataFrame,
) -> Figure:
    """
    Create YOLO training curve visualization
    from results.csv.
    """

    loss_columns = [
        "train/box_loss",
        "train/cls_loss",
        "train/dfl_loss",
        "val/box_loss",
        "val/cls_loss",
        "val/dfl_loss",
    ]

    available = [
        col
        for col in loss_columns
        if col in df.columns
    ]

    if not available:
        raise ValueError(
            f"No YOLO loss columns found.\n"
            f"Available columns: {list(df.columns)}"
        )

    fig = px.line(
        df,
        x="epoch",
        y=available,
        title="Training & Validation Loss Curves",
    )

    fig.update_layout(
        xaxis_title="Epoch",
        yaxis_title="Loss",
        legend_title="Metric",
        height=600,
    )

    return fig