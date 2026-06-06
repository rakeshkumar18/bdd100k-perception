# src/dashboard/visualizations.py

import plotly.express as px


def metric_bar_chart(df, metric):

    chart = px.bar(df, x="run_id", y=metric, title=metric)

    return chart


def create_loss_chart(df):

    loss_cols = [
        "metrics.train/box_loss",
        "metrics.train/cls_loss",
        "metrics.train/dfl_loss",
        "metrics.val/box_loss",
        "metrics.val/cls_loss",
        "metrics.val/dfl_loss",
    ]

    loss_cols = [c for c in loss_cols if c in df.columns]

    melted = df[loss_cols].melt(var_name="Loss", value_name="Value")

    fig = px.bar(melted, x="Loss", y="Value", title="Training & Validation Losses")

    return fig
