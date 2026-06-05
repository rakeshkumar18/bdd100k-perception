# src/dashboard/app.py

from pathlib import Path

import streamlit as st

import pandas as pd
import tempfile



from src.dashboard.mlflow_client import (
    MLflowClient
)

from src.dashboard.visualizations import (
    create_loss_chart
)



st.set_page_config(
    page_title="BDD100K Dashboard",
    layout="wide"
)

page = st.sidebar.selectbox(
    "Page",
    [
        "Training Dashboard",
        "Dataset Analysis",
        "Inference",
        "Evaluation"
    ]
)


def get_metric(row, metric_name):

    candidates = [
        f"metrics.{metric_name}",
        f"metrics.metrics/{metric_name}"
    ]

    for col in candidates:

        if col in row.index:

            value = row[col]

            if pd.notna(value):
                return value

    return None

def get_metric_column(runs, metric_name):

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

def get_latest_valid_run(runs):

    metric_candidates = [
        "metrics.mAP50B",
        "metrics.metrics/mAP50B"
    ]

    for col in metric_candidates:

        if col in runs.columns:

            valid_runs = runs[
                runs[col].notna()
            ]

            if not valid_runs.empty:
                return valid_runs.iloc[0]

    return runs.iloc[0]

st.title("BDD100K YOLO Dashboard")

client = MLflowClient()
runs = client.get_runs()

# ==========================================================
# TRAINING DASHBOARD
# ==========================================================

if page == "Training Dashboard":

    if runs.empty:

        st.warning("No runs found")

    else:

        latest_run = get_latest_valid_run(runs)

        # KPI CARDS

        c1, c2, c3, c4 = st.columns(4)

        map50 = get_metric(
            latest_run,
            "mAP50B"
        )

        map5095 = get_metric(
            latest_run,
            "mAP50-95B"
        )

        precision = get_metric(
            latest_run,
            "precisionB"
        )

        recall = get_metric(
            latest_run,
            "recallB"
        )

        c1.metric(
            "mAP50",
            f"{map50:.3f}" if map50 is not None else "N/A"
        )

        c2.metric(
            "mAP50-95",
            f"{map5095:.3f}" if map5095 is not None else "N/A"
        )

        c3.metric(
            "Precision",
            f"{precision:.3f}" if precision is not None else "N/A"
        )

        c4.metric(
            "Recall",
            f"{recall:.3f}" if recall is not None else "N/A"
        )

        st.divider()

        st.subheader("Run Comparison")

        comparison_df = runs.copy()

        comparison_df["mAP50"] = get_metric_column(
            comparison_df,
            "mAP50B"
        )

        comparison_df["mAP50-95"] = get_metric_column(
            comparison_df,
            "mAP50-95B"
        )

        comparison_df["Precision"] = get_metric_column(
            comparison_df,
            "precisionB"
        )

        comparison_df["Recall"] = get_metric_column(
            comparison_df,
            "recallB"
        )

        comparison_df = comparison_df[
            comparison_df["status"] == "FINISHED"
        ]
        st.dataframe(
            comparison_df[
                [
                    "run_id",
                    "mAP50",
                    "mAP50-95",
                    "Precision",
                    "Recall"
                ]
            ],
            width='stretch'
        )

        st.divider()

        st.subheader("Training Curves")

        st.plotly_chart(
            create_loss_chart(runs),
            width='stretch'
        )

        st.divider()

        st.subheader("Training Configuration")

        param_cols = [
            c for c in runs.columns
            if c.startswith("params.")
        ]

        selected_params = (
            runs[param_cols]
            .T
        )

        st.dataframe(
            selected_params,
            width='stretch'
        )

        # ==================================================
        # TRAINING ARTIFACTS
        # ==================================================

        st.divider()
        st.header("Training Artifacts")

        run_dir = Path(
            "runs/detect/outputs/training/yolov8n_bdd100k"
        )

        results_img = run_dir / "results.png"

        if results_img.exists():

            st.subheader("Training Metrics")

            st.image(
                str(results_img),
                width='stretch'
            )

        st.subheader("Detection Performance")

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                str(run_dir / "BoxPR_curve.png"),
                caption="Precision Recall Curve",
                width='stretch'
            )

        with col2:

            st.image(
                str(run_dir / "BoxF1_curve.png"),
                caption="F1 Curve",
                width='stretch'
            )

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                str(run_dir / "BoxP_curve.png"),
                caption="Precision Curve",
                width='stretch'
            )

        with col2:

            st.image(
                str(run_dir / "BoxR_curve.png"),
                caption="Recall Curve",
                width='stretch'
            )

        st.subheader("Confusion Matrix")

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                str(run_dir / "confusion_matrix.png"),
                caption="Confusion Matrix",
                width='stretch'
            )

        with col2:

            st.image(
                str(run_dir / "confusion_matrix_normalized.png"),
                caption="Normalized Confusion Matrix",
                width='stretch'
            )


# ==========================================================
# DATASET ANALYSIS
# ==========================================================

elif page == "Dataset Analysis":

    st.title("BDD100K Dataset Analysis")

    eda_dir = Path("outputs/figures")

    st.subheader("Class Distribution")

    st.image(
        str(eda_dir / "train_val_class.png"),
        width='stretch'
    )

    st.subheader("Weather Distribution")

    st.image(
        str(eda_dir / "train_val_weather.png"),
        width='stretch'
    )

    st.subheader("Scene Distribution")

    st.image(
        str(eda_dir / "train_val_scene.png"),
        width='stretch'
    )

    st.subheader("Time of Day Distribution")

    st.image(
        str(eda_dir / "train_val_timeofday.png"),
        width='stretch'
    )

# ==========================================================
# Inferencing
# ==========================================================

elif page == "Inference":

    from src.inference.predictor import YOLOPredictor
    from src.inference.utils import extract_detections
    from src.inference.visualize import get_annotated_image

    @st.cache_resource
    def load_predictor():
        return YOLOPredictor(
            model_path="runs/detect/outputs/training/yolov8n_bdd100k/weights/best.pt"
        )

    st.title(
        "YOLO Image Inference"
    )

    predictor = load_predictor()

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        ) as tmp:

            tmp.write(
                uploaded_file.read()
            )

            image_path = tmp.name

        result = predictor.predict(
            image_path
        )

        annotated = (
            get_annotated_image(
                result
            )
        )

        st.image(
            annotated,
            caption="Prediction",
            width='stretch'
        )

        detections = (
            extract_detections(
                result
            )
        )

        st.subheader(
            "Detections"
        )

        if detections:

            df = pd.DataFrame(
                detections
            )

            st.dataframe(
                df,
                width='stretch'
            )

            counts = (
                df["class"]
                .value_counts()
                .reset_index()
            )

            counts.columns = [
                "Class",
                "Count"
            ]

            st.subheader(
                "Object Counts"
            )

            st.dataframe(
                counts,
                width='stretch'
            )

        else:

            st.info(
                "No detections found"
            )

# ==========================================================
# Evaluation
# ==========================================================

elif page == "Evaluation":

    st.title("Model Evaluation")

    if runs.empty:
        st.warning("No runs found")

    else:

        latest_run = get_latest_valid_run(runs)

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "mAP50",
            f"{get_metric(latest_run, 'mAP50B'):.3f}"
        )

        c2.metric(
            "mAP50-95",
            f"{get_metric(latest_run, 'mAP50-95B'):.3f}"
        )

        c3.metric(
            "Precision",
            f"{get_metric(latest_run, 'precisionB'):.3f}"
        )

        c4.metric(
            "Recall",
            f"{get_metric(latest_run, 'recallB'):.3f}"
        )

    st.divider()

    st.subheader("Confusion Matrix")

    run_dir = Path(
        "runs/detect/outputs/training/yolov8n_bdd100k"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            str(run_dir / "confusion_matrix.png"),
            width='stretch'
        )

    with col2:

        st.image(
            str(run_dir / "confusion_matrix_normalized.png"),
            width='stretch'
        )

    st.divider()

    st.subheader("Detection Curves")

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            str(run_dir / "BoxPR_curve.png"),
            width='stretch'
        )

    with col2:

        st.image(
            str(run_dir / "BoxF1_curve.png"),
            width='stretch'
        )