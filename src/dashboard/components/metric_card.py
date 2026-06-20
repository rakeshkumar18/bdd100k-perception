"""Reusable metric card components."""

from typing import Any

import pandas as pd
import streamlit as st


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

        if col not in row.index:
            continue

        value = row[col]

        if pd.notna(value):
            return value

    return None


def display_run_metrics(
    run: pd.Series,
    title: str,
) -> None:
    """Display model metrics."""

    st.subheader(title)

    c1, c2, c3, c4 = st.columns(4)

    map50 = get_metric(
        run,
        "mAP50B",
    )

    map5095 = get_metric(
        run,
        "mAP50-95B",
    )

    precision = get_metric(
        run,
        "precisionB",
    )

    recall = get_metric(
        run,
        "recallB",
    )

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
