"""
Video inference helper functions.
"""

import tempfile
import time
from pathlib import Path

import cv2

from src.inference.predictor import YOLOPredictor

def get_class_counts(
    result,
) -> dict[str, int]:
    """
    Extract class counts from YOLO result.
    """

    class_counts: dict[str, int] = {}

    if result.boxes is None:
        return class_counts

    for cls_id in (
        result.boxes.cls
        .cpu()
        .numpy()
    ):

        class_name = result.names[
            int(cls_id)
        ]

        class_counts[
            class_name
        ] = (
            class_counts.get(
                class_name,
                0,
            )
            + 1
        )

    return class_counts

def save_uploaded_video(
    uploaded_file,
) -> str:
    """
    Save uploaded video to a temporary file.

    Args:
        uploaded_file:
            Streamlit uploaded file.

    Returns:
        Temporary video path.
    """

    suffix = Path(uploaded_file.name).suffix

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as tmp:
        tmp.write(uploaded_file.read())

        return tmp.name

def process_video(
    predictor: YOLOPredictor,
    input_video_path: str,
    output_video_path: str,
    conf: float = 0.25,
    tracking: bool = False,
) -> dict[str, int | float]:
    """
    Run YOLO inference on a video and save
    the annotated output.

    Args:
        predictor:
            YOLO predictor.

        input_video_path:
            Input video path.

        output_video_path:
            Output video path.

        conf:
            Confidence threshold.

        tracking:
            Enable object tracking.

    Returns:
        Processing statistics.
    """

    cap = cv2.VideoCapture(
        input_video_path,
    )

    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {input_video_path}")

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH,
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT,
        )
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS,
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT,
        )
    )

    writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(
            *"mp4v",
        ),
        fps,
        (
            width,
            height,
        ),
    )

    start_time = time.perf_counter()

    processed_frames = 0
    total_detections = 0
    unique_track_ids: set[int] = set()
    class_counts: dict[str, int] = {}

    try:
        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                break

            if tracking:
                result = predictor.track_frame(
                    frame=frame,
                    conf=conf,
                )
            else:
                result = predictor.predict_frame(
                    frame=frame,
                    conf=conf,
                )

            total_detections += len(result.boxes)

            if tracking and result.boxes is not None and result.boxes.id is not None:
                track_ids = result.boxes.id.cpu().numpy().astype(int)

                unique_track_ids.update(track_ids)

            annotated_frame = result.plot()

            writer.write(
                annotated_frame,
            )

            frame_counts = get_class_counts(
                result,
            )

            for cls, count in frame_counts.items():

                class_counts[cls] = (
                    class_counts.get(
                        cls,
                        0,
                    )
                    + count
                )

            processed_frames += 1

    finally:
        cap.release()
        writer.release()

    runtime_seconds = time.perf_counter() - start_time

    processing_fps = processed_frames / runtime_seconds if runtime_seconds > 0 else 0.0

    return {
        "frame_count": frame_count,
        "processed_frames": processed_frames,
        "source_fps": round(
            fps,
            2,
        ),
        "processing_fps": round(
            processing_fps,
            2,
        ),
        "runtime_seconds": round(
            runtime_seconds,
            2,
        ),
        "total_detections": total_detections,
        "unique_tracks": len(
            unique_track_ids,
        ),
        "class_counts": class_counts,
    }

def process_webcam_stream(
    predictor: YOLOPredictor,
    conf: float = 0.25,
):
    """
    Generator yielding annotated webcam frames
    and per-frame class counts.
    """

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise ValueError(
            "Unable to open webcam."
        )

    try:

        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                break

            result = predictor.predict_frame(
                frame=frame,
                conf=conf,
            )

            annotated_frame = result.plot()

            class_counts = get_class_counts(
                result,
            )

            yield (
                annotated_frame,
                class_counts,
            )

    finally:


        cap.release()