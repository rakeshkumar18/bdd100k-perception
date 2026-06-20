import av
import streamlit as st

from streamlit_webrtc import (
    VideoProcessorBase,
    webrtc_streamer,
)

from src.dashboard.components.inference_utils import (
    load_predictor,
)

predictor = load_predictor()


"""
Webcam inference using streamlit-webrtc.
"""

import av

from streamlit_webrtc import VideoProcessorBase

from src.inference.predictor import YOLOPredictor


class YOLOWebcamProcessor(VideoProcessorBase):
    """
    WebRTC video processor for YOLO webcam inference.
    """

    def __init__(self) -> None:
        self.predictor = None

    def set_predictor(self, predictor: YOLOPredictor) -> None:
        self.predictor = predictor

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        result = self.predictor.predict_frame(
            frame=image,
            conf=0.25,
        )

        annotated_frame = result.plot()

        class_counts: dict[str, int] = {}

        if result.boxes is not None:

            for cls_id in result.boxes.cls.cpu().numpy():

                class_name = result.names[int(cls_id)]

                class_counts[class_name] = (
                    class_counts.get(class_name, 0) + 1
                )

        return av.VideoFrame.from_ndarray(
            annotated_frame,
            format="bgr24",
        ), class_counts