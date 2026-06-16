"""Inference dashboard page."""

import tempfile

import pandas as pd
import streamlit as st

from src.dashboard.components.inference_utils import (
    extract_detections,
    get_annotated_image,
    load_predictor,
)


# ==========================================================
# HELPERS
# ==========================================================


def show_detection_table(
    detections: list[dict],
) -> pd.DataFrame | None:
    """Display detection table."""

    if not detections:
        st.info(
            "No detections found."
        )
        return None

    df = pd.DataFrame(
        detections
    )

    st.dataframe(
        df,
        width="stretch",
    )

    return df


def show_class_counts(
    detections_df: pd.DataFrame,
) -> pd.DataFrame:
    """Display object counts."""

    counts = (
        detections_df["class"]
        .value_counts()
        .reset_index()
    )

    counts.columns = [
        "Class",
        "Count",
    ]

    st.dataframe(
        counts,
        width="stretch",
    )

    return counts


# ==========================================================
# PAGE
# ==========================================================


def render() -> None:
    """Render inference page."""

    st.title(
        "YOLO Multi-Image Inference"
    )

    uploaded_files = st.file_uploader(
        "Upload Images",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        return

    predictor = load_predictor()

    total_counts: dict[str, int] = {}

    for uploaded_file in uploaded_files:

        st.divider()

        st.subheader(
            uploaded_file.name
        )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg",
        ) as tmp:

            tmp.write(
                uploaded_file.read()
            )

            image_path = tmp.name

        result = predictor.predict(
            image_path
        )

        annotated_image = (
            get_annotated_image(
                result
            )
        )

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image_path,
                caption="Original Image",
                width="stretch",
            )

        with col2:
            st.image(
                annotated_image,
                caption="YOLO Prediction",
                width="stretch",
            )

        detections = (
            extract_detections(
                result
            )
        )

        st.subheader(
            "Detections"
        )

        detections_df = (
            show_detection_table(
                detections
            )
        )

        if detections_df is None:
            continue

        st.subheader(
            "Object Counts"
        )

        counts_df = (
            show_class_counts(
                detections_df
            )
        )

        for _, row in counts_df.iterrows():

            cls = row["Class"]

            total_counts[cls] = (
                total_counts.get(
                    cls,
                    0,
                )
                + int(
                    row["Count"]
                )
            )

    st.divider()

    st.header(
        "Overall Detection Summary"
    )

    if not total_counts:
        st.info(
            "No detections across uploaded images."
        )
        return

    summary_df = (
        pd.DataFrame(
            {
                "Class": total_counts.keys(),
                "Count": total_counts.values(),
            }
        )
        .sort_values(
            by="Count",
            ascending=False,
        )
    )

    st.dataframe(
        summary_df,
        width="stretch",
    )

    st.bar_chart(
        summary_df.set_index(
            "Class"
        )
    )
