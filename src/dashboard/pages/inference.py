"""Inference dashboard page."""

import tempfile

import pandas as pd
import streamlit as st

from src.arduino.controller import ArduinoController
from src.logic.event_mapper import map_event

from src.dashboard.components.inference_utils import (
    extract_detections,
    get_annotated_image,
    load_predictor,
)
from src.inference.events import (
    EventGenerator
)

from src.dashboard.components.video_utils import (
    process_video,
    save_uploaded_video,process_webcam_stream
)

# ==========================================================
# HELPERS
# ==========================================================


def show_detection_table(
    detections: list[dict],
) -> pd.DataFrame | None:
    """Display detection table."""

    if not detections:
        st.info("No detections found.")
        return None

    df = pd.DataFrame(detections)

    st.dataframe(
        df,
        width="stretch",
    )

    return df


def show_class_counts(
    detections_df: pd.DataFrame,
) -> pd.DataFrame:
    """Display object counts."""

    counts = detections_df["class"].value_counts().reset_index()

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


def render_image_inference() -> None:
    """Render inference page."""

    uploaded_files = st.file_uploader(
        "Upload Images",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
        accept_multiple_files=True,
        key="image_uploader",
    )

    if not uploaded_files:
        return

    predictor = load_predictor()

    total_counts: dict[str, int] = {}

    for uploaded_file in uploaded_files:

        st.divider()

        st.subheader(uploaded_file.name)

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg",
        ) as tmp:

            tmp.write(uploaded_file.read())

            image_path = tmp.name

        result = predictor.predict(image_path)

        annotated_image = get_annotated_image(result)

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

        detections = extract_detections(result)

        st.subheader("Detections")

        detections_df = show_detection_table(detections)

        if detections_df is None:
            continue

        st.subheader("Object Counts")

        counts_df = show_class_counts(detections_df)

        for _, row in counts_df.iterrows():

            cls = row["Class"]

            total_counts[cls] = total_counts.get(
                cls,
                0,
            ) + int(row["Count"])

    st.divider()

    st.header("Overall Detection Summary")

    if not total_counts:
        st.info("No detections across uploaded images.")
        return

    summary_df = pd.DataFrame(
        {
            "Class": total_counts.keys(),
            "Count": total_counts.values(),
        }
    ).sort_values(
        by="Count",
        ascending=False,
    )

    st.dataframe(
        summary_df,
        width="stretch",
    )

    st.bar_chart(summary_df.set_index("Class"))


def render_video_inference() -> None:
    """Render video inference."""

    uploaded_video = st.file_uploader(
        "Upload Video",
        type=[
            "mp4",
            "mov",
            "avi",
        ],
        key="video_uploader",
    )

    if "event_generator" not in st.session_state:
        st.session_state.event_generator = EventGenerator()

    generate_events = st.session_state.event_generator

    if uploaded_video is None:
        return

    enable_tracking = st.checkbox(
        "Enable Tracking",
        value=False,
        key="enable_video_tracking",
    )
    enable_arduino = st.checkbox(
        "Enable Arduino Controller",
        value=False,
        key="enable_arduino",
    )

    if st.button(
        "Run Video Inference",
        key="run_video_inference",
        type="primary",
    ):
        predictor = load_predictor()

        input_video_path = save_uploaded_video(
            uploaded_video,
        )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4",
        ) as tmp:
            output_video_path = tmp.name

        with st.spinner(
            "Running inference...",
        ):
            stats = process_video(
                predictor=predictor,
                input_video_path=input_video_path,
                output_video_path=output_video_path,
                tracking=enable_tracking,
            )

        st.success("Inference completed.")

        st.subheader("Processing Statistics")

        # --------------------------------------------------
        # Metrics Row 1
        # --------------------------------------------------

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Frames",
                stats["processed_frames"],
            )

        with col2:
            st.metric(
                "Detections",
                stats["total_detections"],
            )

        with col3:
            st.metric(
                "Processing FPS",
                stats["processing_fps"],
            )

        # --------------------------------------------------
        # Metrics Row 2
        # --------------------------------------------------

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Source FPS",
                stats["source_fps"],
            )

        with col2:
            st.metric(
                "Runtime (s)",
                stats["runtime_seconds"],
            )

        with col3:
            if enable_tracking:
                st.metric(
                    "Tracks",
                    stats["unique_tracks"],
                )

        st.subheader("Annotated Video")

        st.video(output_video_path)

        st.subheader("Object Analytics")

        analytics_df = pd.DataFrame(
            {
                "Class": stats["class_counts"].keys(),
                "Count": stats["class_counts"].values(),
            }
        ).sort_values(
            by="Count",
            ascending=False,
        )

        if not analytics_df.empty:
            top_class = analytics_df.iloc[0]["Class"]
        else:
            top_class = "N/A"

        unique_classes = len(analytics_df)

        avg_detections = (
            round(
                stats["total_detections"] / stats["processed_frames"],
                2,
            )
            if stats["processed_frames"] > 0
            else 0.0
        )
        st.subheader("Analytics Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Top Object",
                top_class,
            )

        with col2:
            st.metric(
                "Unique Classes",
                unique_classes,
            )

        with col3:
            st.metric(
                "Detections / Frame",
                avg_detections,
            )
        st.dataframe(
            analytics_df,
            width="stretch",
        )

        csv_data = analytics_df.to_csv(
            index=False,
        )

        st.download_button(
            label="Download Analytics CSV",
            data=csv_data,
            file_name="video_analytics.csv",
            mime="text/csv",
        )

        st.bar_chart(
            analytics_df.set_index(
                "Class",
            )
        )

        events = generate_events.generate(stats["class_counts"])

        if enable_arduino and events:

            try:
                with ArduinoController() as arduino:

                    for event in events:

                        command = map_event(event)

                        if command:
                            arduino.send(command)

            except Exception as exc:
                st.error(
                    f"Arduino communication failed: {exc}"
                )

        st.subheader("Detected Events")

        if events:
            for event in events:
                st.success(event)
        else:
            st.info("No events generated.")

def render_webcam_inference() -> None:
    """Render live webcam inference."""

    st.subheader(
        "Live Webcam Detection"
    )

    start_stream = st.button(
        "Start Webcam",
        type="primary",
    )

    if not start_stream:
        return

    predictor = load_predictor()

    event_generator = EventGenerator()

    frame_placeholder = st.empty()

    event_placeholder = st.empty()

    try:

        with ArduinoController() as arduino:

            last_pedestrian_state = False

            for (
                annotated_frame,
                class_counts,
            ) in process_webcam_stream(
                predictor=predictor,
            ):

                frame_placeholder.image(
                    annotated_frame,
                    channels="BGR",
                    width="stretch",
                )

                events = (
                    event_generator.generate(
                        class_counts
                    )
                )

                pedestrian_detected = (
                    "PEDESTRIAN_DETECTED"
                    in events
                )

                if (
                    pedestrian_detected
                    and not last_pedestrian_state
                ):

                    command = map_event(
                        "PEDESTRIAN_DETECTED"
                    )

                    if command:
                        arduino.send(
                            command
                        )

                elif (
                    not pedestrian_detected
                    and last_pedestrian_state
                ):

                    arduino.send(
                        "OFF"
                    )

                last_pedestrian_state = (
                    pedestrian_detected
                )

                event_placeholder.empty()

                for event in events:
                    event_placeholder.success(
                        event
                    )

    except Exception as exc:

        st.error(
            f"Arduino communication failed: {exc}"
        )

def render() -> None:
    """Render inference page."""

    st.title("YOLO Inference")

    image_tab, video_tab, webcam_tab = st.tabs(
        [
            "📷 Images",
            "🎥 Videos",
            "📹 Live Webcam",
        ]
    )

    with image_tab:
            render_image_inference()

    with video_tab:
        render_video_inference()

    with webcam_tab:
        render_webcam_inference()
