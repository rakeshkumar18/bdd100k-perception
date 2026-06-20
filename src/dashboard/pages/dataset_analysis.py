"""Dataset analysis dashboard page."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.utils.paths import (
    FIGURES_DIR,
    REPORTS_DIR,
)
from src.dashboard.components.file_viewers import (
    show_image,
    show_table,
)


def render() -> None:
    """Render dataset analysis page."""

    st.title("BDD100K Dataset Analysis")

    st.markdown("""
        This section analyzes the **BDD100K object detection dataset**.

        ### Included

        - bike
        - bus
        - car
        - motor
        - person
        - rider
        - traffic light
        - traffic sign
        - train
        - truck

        ### Included Splits

        - Train
        - Validation

        ### Excluded

        - Test
        - Lane Marking
        - Drivable Area
        - Semantic Segmentation
        """)

    figures_dir = FIGURES_DIR
    reports_dir = REPORTS_DIR

    # ======================================================
    # SUMMARY
    # ======================================================

    st.subheader("Dataset Summary")

    col1, col2 = st.columns(2)

    with col1:
        show_table(
            "train_class_distribution.csv",
            "Train Class Distribution",
            reports_dir,
        )

    with col2:
        show_table(
            "val_class_distribution.csv",
            "Validation Class Distribution",
            reports_dir,
        )

    # ======================================================
    # TABS
    # ======================================================

    (
        tab1,
        tab2,
        tab3,
        tab4,
    ) = st.tabs(
        [
            "Train vs Val",
            "Scene Attributes",
            "Box & Occlusion",
            "Data Quality",
        ]
    )

    # ======================================================
    # TRAIN VS VAL
    # ======================================================

    with tab1:

        st.markdown("### Train vs Validation Comparison")

        for image_file, caption in [
            (
                "train_val_class.png",
                "Train vs Validation Class Distribution",
            ),
            (
                "train_val_weather.png",
                "Train vs Validation Weather Distribution",
            ),
            (
                "train_val_scene.png",
                "Train vs Validation Scene Distribution",
            ),
            (
                "train_val_timeofday.png",
                "Train vs Validation Time of Day Distribution",
            ),
        ]:
            show_image(
                image_file,
                caption,
                figures_dir,
            )

        for csv_file, title in [
            (
                "train_val_class.csv",
                "Train vs Validation Class Table",
            ),
            (
                "train_val_weather.csv",
                "Train vs Validation Weather Table",
            ),
            (
                "train_val_scene.csv",
                "Train vs Validation Scene Table",
            ),
            (
                "train_val_timeofday.csv",
                "Train vs Validation Time-of-Day Table",
            ),
        ]:
            show_table(
                csv_file,
                title,
                reports_dir,
            )

    # ======================================================
    # SCENE ATTRIBUTES
    # ======================================================

    with tab2:

        st.markdown("### Scene, Weather, and Time-of-Day Analysis")

        split = st.radio(
            "Select Split",
            ["train", "val"],
            horizontal=True,
            key="scene_split",
        )

        for image_file, caption in [
            (
                f"{split}_weather_distribution.png",
                f"{split.title()} Weather Distribution",
            ),
            (
                f"{split}_scene_distribution.png",
                f"{split.title()} Scene Distribution",
            ),
            (
                f"{split}_timeofday_distribution.png",
                f"{split.title()} Time-of-Day Distribution",
            ),
        ]:
            show_image(
                image_file,
                caption,
                figures_dir,
            )

        for csv_file, title in [
            (
                f"{split}_weather_distribution.csv",
                f"{split.title()} Weather Table",
            ),
            (
                f"{split}_scene_distribution.csv",
                f"{split.title()} Scene Table",
            ),
            (
                f"{split}_timeofday_distribution.csv",
                f"{split.title()} Time-of-Day Table",
            ),
        ]:
            show_table(
                csv_file,
                title,
                reports_dir,
            )

    # ======================================================
    # BOX ANALYSIS
    # ======================================================

    with tab3:

        st.markdown("### Bounding Box and Occlusion Analysis")

        split = st.radio(
            "Select Split",
            ["train", "val"],
            horizontal=True,
            key="bbox_split",
        )

        for image_file, caption in [
            (
                f"{split}_class_distribution.png",
                f"{split.title()} Class Distribution",
            ),
            (
                f"{split}_bbox_area_histogram.png",
                f"{split.title()} Bounding Box Area Histogram",
            ),
            (
                f"{split}_bbox_area_log_histogram.png",
                f"{split.title()} Log Bounding Box Area Histogram",
            ),
            (
                f"{split}_aspect_ratio_histogram.png",
                f"{split.title()} Aspect Ratio Histogram",
            ),
            (
                f"{split}_occlusion_distribution.png",
                f"{split.title()} Occlusion Distribution",
            ),
            (
                f"{split}_occlusion_by_class.png",
                f"{split.title()} Occlusion by Class",
            ),
        ]:
            show_image(
                image_file,
                caption,
                figures_dir,
            )

    # ======================================================
    # DATA QUALITY
    # ======================================================

    with tab4:

        st.markdown("### Data Quality Checks")

        st.info(
            "Checks include:\n"
            "- Zero-area bounding boxes\n"
            "- Negative-area bounding boxes\n"
            "- Zero width or height\n"
            "- Negative width or height"
        )

        quality_train = reports_dir / "train_quality_report.csv"

        quality_val = reports_dir / "val_quality_report.csv"

        invalid_boxes = reports_dir / "invalid_boxes.csv"

        if quality_train.exists():
            st.markdown("#### Train Quality Report")
            st.dataframe(
                pd.read_csv(quality_train),
                width="stretch",
            )

        if quality_val.exists():
            st.markdown("#### Validation Quality Report")
            st.dataframe(
                pd.read_csv(quality_val),
                width="stretch",
            )

        if invalid_boxes.exists():
            st.markdown("#### Invalid Bounding Boxes")
            st.dataframe(
                pd.read_csv(invalid_boxes),
                width="stretch",
            )

        if (
            not quality_train.exists()
            and not quality_val.exists()
            and not invalid_boxes.exists()
        ):
            st.warning("No quality reports found.")
