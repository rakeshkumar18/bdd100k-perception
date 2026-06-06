"""Run the Streamlit dashboard for BDD100K training, analysis, inference, and evaluation."""

from pathlib import Path
import tempfile
from typing import Any

import pandas as pd
import streamlit as st

from src.dashboard.mlflow_client import MLflowClient
from src.dashboard.visualizations import create_loss_chart
from src.inference.predictor import YOLOPredictor
from src.inference.utils import extract_detections
from src.inference.visualize import get_annotated_image


@st.cache_resource
def load_predictor() -> YOLOPredictor:
    """Load and cache the YOLO predictor for dashboard inference."""
    return YOLOPredictor(
        model_path="runs/detect/outputs/training/yolov8n_bdd100k/weights/best.pt"
    )


st.set_page_config(page_title="BDD100K Dashboard", layout="wide")

page = st.sidebar.selectbox(
    "Page", ["Training Dashboard", "Dataset Analysis", "Inference", "Evaluation"]
)


def get_metric(row: pd.Series, metric_name: str) -> Any:
    """Return a metric value from an MLflow run row using known column patterns.

    Args:
        row: A pandas Series representing one MLflow run.
        metric_name: Metric suffix to look up, such as ``mAP50B``.

    Returns:
        The metric value if available, otherwise ``None``.
    """
    candidates = [f"metrics.{metric_name}", f"metrics.metrics/{metric_name}"]

    for col in candidates:
        if col in row.index:
            value = row[col]
            if pd.notna(value):
                return value

    return None


def get_metric_column(runs: pd.DataFrame, metric_name: str) -> pd.Series | None:
    """Return a metric column from the MLflow runs table.

    Args:
        runs: DataFrame containing MLflow run metadata.
        metric_name: Metric suffix to look up.

    Returns:
        A pandas Series containing the metric values, or ``None`` if unavailable.
    """
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


def get_latest_valid_run(runs: pd.DataFrame) -> pd.Series:
    """Return the most recent run with valid evaluation metrics.

    Args:
        runs: DataFrame containing MLflow runs.

    Returns:
        A pandas Series representing the latest valid run.
    """
    metric_candidates = ["metrics.mAP50B", "metrics.metrics/mAP50B"]

    for col in metric_candidates:
        if col in runs.columns:
            valid_runs = runs[runs[col].notna()]

            if not valid_runs.empty:
                return valid_runs.iloc[0]

    return runs.iloc[0]


def show_image(image_name: str, caption: str, image_dir: Path) -> None:
    """Display an image from disk if it exists.

    Args:
        image_name: Name of the image file.
        caption: Caption displayed below the image.
        image_dir: Directory containing dashboard figure assets.
    """
    image_path = image_dir / image_name
    if image_path.exists():
        st.image(str(image_path), caption=caption, width="stretch")
    else:
        st.warning(f"Missing figure: {image_path}")


def show_table(csv_name: str, title: str, table_dir: Path) -> None:
    """Display a CSV file as a Streamlit table if it exists.

    Args:
        csv_name: Name of the CSV file.
        title: Table section title.
        table_dir: Directory containing report tables.
    """
    csv_path = table_dir / csv_name
    if csv_path.exists():
        st.markdown(f"#### {title}")
        df = pd.read_csv(csv_path)
        st.dataframe(df, width="stretch")
    else:
        st.info(f"Missing table: {csv_path}")


st.title("BDD100K YOLO Dashboard")

client = MLflowClient()
runs = client.get_runs()


if page == "Training Dashboard":
    if runs.empty:
        st.warning("No runs found")
    else:
        latest_run = get_latest_valid_run(runs)

        c1, c2, c3, c4 = st.columns(4)

        map50 = get_metric(latest_run, "mAP50B")
        map5095 = get_metric(latest_run, "mAP50-95B")
        precision = get_metric(latest_run, "precisionB")
        recall = get_metric(latest_run, "recallB")

        c1.metric("mAP50", f"{map50:.3f}" if map50 is not None else "N/A")
        c2.metric("mAP50-95", f"{map5095:.3f}" if map5095 is not None else "N/A")
        c3.metric("Precision", f"{precision:.3f}" if precision is not None else "N/A")
        c4.metric("Recall", f"{recall:.3f}" if recall is not None else "N/A")

        st.divider()
        st.subheader("Run Comparison")

        comparison_df = runs.copy()
        comparison_df["mAP50"] = get_metric_column(comparison_df, "mAP50B")
        comparison_df["mAP50-95"] = get_metric_column(comparison_df, "mAP50-95B")
        comparison_df["Precision"] = get_metric_column(comparison_df, "precisionB")
        comparison_df["Recall"] = get_metric_column(comparison_df, "recallB")

        comparison_df = comparison_df[comparison_df["status"] == "FINISHED"]
        st.dataframe(
            comparison_df[["run_id", "mAP50", "mAP50-95", "Precision", "Recall"]],
            width="stretch",
        )

        st.divider()
        st.subheader("Training Curves")
        st.plotly_chart(create_loss_chart(runs), width="stretch")

        st.divider()
        st.subheader("Training Configuration")

        param_cols = [col for col in runs.columns if col.startswith("params.")]
        selected_params = runs[param_cols].T
        st.dataframe(selected_params, width="stretch")

        st.divider()
        st.header("Training Artifacts")

        run_dir = Path("runs/detect/outputs/training/yolov8n_bdd100k")
        results_img = run_dir / "results.png"

        if results_img.exists():
            st.subheader("Training Metrics")
            st.image(str(results_img), width="stretch")

        st.subheader("Detection Performance")

        col1, col2 = st.columns(2)
        with col1:
            st.image(
                str(run_dir / "BoxPR_curve.png"),
                caption="Precision Recall Curve",
                width="stretch",
            )
        with col2:
            st.image(
                str(run_dir / "BoxF1_curve.png"),
                caption="F1 Curve",
                width="stretch",
            )

        col1, col2 = st.columns(2)
        with col1:
            st.image(
                str(run_dir / "BoxP_curve.png"),
                caption="Precision Curve",
                width="stretch",
            )
        with col2:
            st.image(
                str(run_dir / "BoxR_curve.png"),
                caption="Recall Curve",
                width="stretch",
            )

        st.subheader("Confusion Matrix")

        col1, col2 = st.columns(2)
        with col1:
            st.image(
                str(run_dir / "confusion_matrix.png"),
                caption="Confusion Matrix",
                width="stretch",
            )
        with col2:
            st.image(
                str(run_dir / "confusion_matrix_normalized.png"),
                caption="Normalized Confusion Matrix",
                width="stretch",
            )

elif page == "Dataset Analysis":
    st.title("BDD100K Dataset Analysis")

    st.markdown("""
        This section analyzes the **BDD100K object detection** dataset only.

        **Included**
        - 10 detection classes with `box2d` annotations:
          `bike`, `bus`, `car`, `motor`, `person`, `rider`,
          `traffic light`, `traffic sign`, `train`, `truck`
        - Train split
        - Validation split

        **Excluded**
        - Test split
        - Lane marking
        - Drivable area
        - Semantic segmentation
        """)

    eda_dir = Path("outputs/figures")
    report_dir = Path("outputs/reports")

    st.subheader("Dataset Summary")

    col1, col2 = st.columns(2)
    with col1:
        show_table(
            "train_class_distribution.csv",
            "Train Class Distribution",
            report_dir,
        )
    with col2:
        show_table(
            "val_class_distribution.csv",
            "Validation Class Distribution",
            report_dir,
        )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Train vs Val", "Scene Attributes", "Box and Occlusion", "Data Quality"]
    )

    with tab1:
        st.markdown("### Train vs Validation Comparison")
        show_image(
            "train_val_class.png",
            "Train vs Validation Class Distribution",
            eda_dir,
        )
        show_image(
            "train_val_weather.png",
            "Train vs Validation Weather Distribution",
            eda_dir,
        )
        show_image(
            "train_val_scene.png",
            "Train vs Validation Scene Distribution",
            eda_dir,
        )
        show_image(
            "train_val_timeofday.png",
            "Train vs Validation Time of Day Distribution",
            eda_dir,
        )

        show_table("train_val_class.csv", "Train vs Validation Class Table", report_dir)
        show_table(
            "train_val_weather.csv",
            "Train vs Validation Weather Table",
            report_dir,
        )
        show_table("train_val_scene.csv", "Train vs Validation Scene Table", report_dir)
        show_table(
            "train_val_timeofday.csv",
            "Train vs Validation Time of Day Table",
            report_dir,
        )

    with tab2:
        st.markdown("### Scene, Weather, and Time-of-Day Analysis")

        split_option = st.radio(
            "Select split",
            ["train", "val"],
            horizontal=True,
            key="scene_split",
        )

        show_image(
            f"{split_option}_weather_distribution.png",
            f"{split_option.title()} Weather Distribution",
            eda_dir,
        )
        show_image(
            f"{split_option}_scene_distribution.png",
            f"{split_option.title()} Scene Distribution",
            eda_dir,
        )
        show_image(
            f"{split_option}_timeofday_distribution.png",
            f"{split_option.title()} Time-of-Day Distribution",
            eda_dir,
        )

        show_table(
            f"{split_option}_weather_distribution.csv",
            f"{split_option.title()} Weather Table",
            report_dir,
        )
        show_table(
            f"{split_option}_scene_distribution.csv",
            f"{split_option.title()} Scene Table",
            report_dir,
        )
        show_table(
            f"{split_option}_timeofday_distribution.csv",
            f"{split_option.title()} Time-of-Day Table",
            report_dir,
        )

    with tab3:
        st.markdown("### Bounding Box and Occlusion Analysis")

        split_option = st.radio(
            "Select split ",
            ["train", "val"],
            horizontal=True,
            key="bbox_split",
        )

        show_image(
            f"{split_option}_class_distribution.png",
            f"{split_option.title()} Class Distribution",
            eda_dir,
        )
        show_image(
            f"{split_option}_bbox_area_histogram.png",
            f"{split_option.title()} Bounding Box Area Histogram",
            eda_dir,
        )
        show_image(
            f"{split_option}_bbox_area_log_histogram.png",
            f"{split_option.title()} Log Bounding Box Area Histogram",
            eda_dir,
        )
        show_image(
            f"{split_option}_aspect_ratio_histogram.png",
            f"{split_option.title()} Aspect Ratio Histogram",
            eda_dir,
        )
        show_image(
            f"{split_option}_occlusion_distribution.png",
            f"{split_option.title()} Occlusion Distribution",
            eda_dir,
        )
        show_image(
            f"{split_option}_occlusion_by_class.png",
            f"{split_option.title()} Occlusion by Class",
            eda_dir,
        )

    with tab4:
        st.markdown("### Data Quality Checks")

        st.info(
            "Recommended checks from the analysis script:\n"
            "- Zero-area bounding boxes\n"
            "- Negative-area bounding boxes\n"
            "- Zero width or height\n"
            "- Negative width or height"
        )

        quality_train = report_dir / "train_quality_report.csv"
        quality_val = report_dir / "invalid_boxes.csv"

        if quality_train.exists():
            st.markdown("#### Train Quality Report")
            st.dataframe(pd.read_csv(quality_train), width="stretch")

        if quality_val.exists():
            st.markdown("#### Validation Quality Report")
            st.dataframe(pd.read_csv(quality_val), width="stretch")

        if not quality_train.exists() and not quality_val.exists():
            st.warning(
                "No saved quality report CSV found yet. "
                "You can export the data quality results to CSV and display them here."
            )

elif page == "Inference":
    st.title("YOLO Multi-Image Inference")

    predictor = load_predictor()

    uploaded_files = st.file_uploader(
        "Upload Images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        total_counts: dict[str, int] = {}

        for uploaded_file in uploaded_files:
            st.divider()
            st.subheader(uploaded_file.name)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(uploaded_file.read())
                image_path = tmp.name

            result = predictor.predict(image_path)
            annotated = get_annotated_image(result)

            col1, col2 = st.columns(2)
            with col1:
                st.image(image_path, caption="Original Image", width="stretch")
            with col2:
                st.image(annotated, caption="YOLO Prediction", width="stretch")

            detections = extract_detections(result)

            st.subheader("Detections")

            if detections:
                df = pd.DataFrame(detections)
                st.dataframe(df, width="stretch")

                counts = df["class"].value_counts().reset_index()
                counts.columns = ["Class", "Count"]

                st.subheader("Object Counts")
                st.dataframe(counts, width="stretch")

                for _, row in counts.iterrows():
                    cls = row["Class"]
                    cnt = row["Count"]
                    total_counts[cls] = total_counts.get(cls, 0) + cnt
            else:
                st.info("No detections found")

        st.divider()
        st.header("Overall Detection Summary")

        if total_counts:
            summary_df = pd.DataFrame(
                {"Class": total_counts.keys(), "Count": total_counts.values()}
            )
            summary_df = summary_df.sort_values(by="Count", ascending=False)

            st.dataframe(summary_df, width="stretch")
            st.bar_chart(summary_df.set_index("Class"))

elif page == "Evaluation":
    st.title("Model Evaluation")

    if runs.empty:
        st.warning("No runs found")
    else:
        latest_run = get_latest_valid_run(runs)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("mAP50", f"{get_metric(latest_run, 'mAP50B'):.3f}")
        c2.metric("mAP50-95", f"{get_metric(latest_run, 'mAP50-95B'):.3f}")
        c3.metric("Precision", f"{get_metric(latest_run, 'precisionB'):.3f}")
        c4.metric("Recall", f"{get_metric(latest_run, 'recallB'):.3f}")

    st.divider()
    st.subheader("Confusion Matrix")

    run_dir = Path("runs/detect/outputs/training/yolov8n_bdd100k")

    col1, col2 = st.columns(2)
    with col1:
        st.image(str(run_dir / "confusion_matrix.png"), width="stretch")
    with col2:
        st.image(str(run_dir / "confusion_matrix_normalized.png"), width="stretch")

    st.divider()
    st.subheader("Detection Curves")

    col1, col2 = st.columns(2)
    with col1:
        st.image(str(run_dir / "BoxPR_curve.png"), width="stretch")
    with col2:
        st.image(str(run_dir / "BoxF1_curve.png"), width="stretch")
