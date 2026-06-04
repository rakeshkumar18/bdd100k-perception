# src/dashboard/app.py

from pathlib import Path

import streamlit as st

import pandas as pd
import tempfile

from src.inference.predictor import (
    YOLOPredictor
)

from src.inference.utils import (
    extract_detections
)

from src.inference.visualize import (
    get_annotated_image
)

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
@st.cache_resource
def load_predictor():

    return YOLOPredictor(
        model_path=
        "runs/detect/outputs/training/yolov8n_bdd100k/weights/best.pt"
    )

page = st.sidebar.selectbox(
    "Page",
    [
        "Training Dashboard",
        "Dataset Analysis",
        "Inference"
    ]
)


def get_metric(row, metric_name):

    column = f"metrics.metrics/{metric_name}"

    if column in row.index:
        return row[column]

    return None


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

        latest_run = runs.iloc[0]

        # KPI CARDS

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "mAP50",
            round(get_metric(latest_run, "mAP50B") or 0, 3)
        )

        c2.metric(
            "mAP50-95",
            round(get_metric(latest_run, "mAP50-95B") or 0, 3)
        )

        c3.metric(
            "Precision",
            round(get_metric(latest_run, "precisionB") or 0, 3)
        )

        c4.metric(
            "Recall",
            round(get_metric(latest_run, "recallB") or 0, 3)
        )

        st.divider()

        st.subheader("Run Comparison")

        comparison_cols = [
            "run_id",
            "metrics.metrics/mAP50B",
            "metrics.metrics/mAP50-95B",
            "metrics.metrics/precisionB",
            "metrics.metrics/recallB",
        ]

        available_cols = [
            c for c in comparison_cols
            if c in runs.columns
        ]

        st.dataframe(
            runs[available_cols],
            use_container_width=True
        )

        st.divider()

        st.subheader("Training Curves")

        st.plotly_chart(
            create_loss_chart(runs),
            use_container_width=True
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
            use_container_width=True
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
                use_container_width=True
            )

        st.subheader("Detection Performance")

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                str(run_dir / "BoxPR_curve.png"),
                caption="Precision Recall Curve",
                use_container_width=True
            )

        with col2:

            st.image(
                str(run_dir / "BoxF1_curve.png"),
                caption="F1 Curve",
                use_container_width=True
            )

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                str(run_dir / "BoxP_curve.png"),
                caption="Precision Curve",
                use_container_width=True
            )

        with col2:

            st.image(
                str(run_dir / "BoxR_curve.png"),
                caption="Recall Curve",
                use_container_width=True
            )

        st.subheader("Confusion Matrix")

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                str(run_dir / "confusion_matrix.png"),
                caption="Confusion Matrix",
                use_container_width=True
            )

        with col2:

            st.image(
                str(run_dir / "confusion_matrix_normalized.png"),
                caption="Normalized Confusion Matrix",
                use_container_width=True
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
        use_container_width=True
    )

    st.subheader("Weather Distribution")

    st.image(
        str(eda_dir / "train_val_weather.png"),
        use_container_width=True
    )

    st.subheader("Scene Distribution")

    st.image(
        str(eda_dir / "train_val_scene.png"),
        use_container_width=True
    )

    st.subheader("Time of Day Distribution")

    st.image(
        str(eda_dir / "train_val_timeofday.png"),
        use_container_width=True
    )

# ==========================================================
# Inferencing
# ==========================================================

elif page == "Inference":

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
            use_container_width=True
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
                use_container_width=True
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
                use_container_width=True
            )

        else:

            st.info(
                "No detections found"
            )