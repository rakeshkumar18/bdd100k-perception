"""
Tests for video inference utilities.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.dashboard.components.video_utils import (
    process_video,
    save_uploaded_video,
)


class MockTensor:
    """Mock tensor."""

    def cpu(self):
        return self

    def numpy(self):
        return np.array(
            [
                0,
                1,
            ]
        )


class MockBoxes:
    """Mock YOLO boxes."""

    def __init__(
        self,
    ) -> None:
        self.id = None

        self.cls = MockTensor()

    def __len__(
        self,
    ) -> int:
        return 2


class MockResult:
    """Mock YOLO result."""

    def __init__(
        self,
    ) -> None:
        self.boxes = MockBoxes()

        self.names = {
            0: "car",
            1: "person",
        }

    def plot(
        self,
    ) -> np.ndarray:
        return np.zeros(
            (
                480,
                640,
                3,
            ),
            dtype=np.uint8,
        )


class MockPredictor:
    """Mock predictor."""

    def predict_frame(
        self,
        frame,
        conf: float = 0.25,
    ):
        return MockResult()

    def track_frame(
        self,
        frame,
        conf: float = 0.25,
    ):
        return MockResult()


class MockUpload:
    """Mock Streamlit upload."""

    def __init__(
        self,
        content: bytes,
    ) -> None:
        self.name = "test.mp4"

        self._content = content

    def read(
        self,
    ) -> bytes:
        return self._content


@pytest.fixture
def sample_video(
    tmp_path: Path,
) -> Path:
    """Create temporary video."""

    video_path = tmp_path / "input.mp4"

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(
            *"mp4v",
        ),
        10,
        (
            640,
            480,
        ),
    )

    for _ in range(5):
        frame = np.zeros(
            (
                480,
                640,
                3,
            ),
            dtype=np.uint8,
        )

        writer.write(
            frame,
        )

    writer.release()

    return video_path


def test_process_video(
    sample_video: Path,
    tmp_path: Path,
) -> None:
    """Test video processing."""

    output_path = tmp_path / "output.mp4"

    stats = process_video(
        predictor=MockPredictor(),
        input_video_path=str(
            sample_video,
        ),
        output_video_path=str(
            output_path,
        ),
    )

    assert output_path.exists()

    assert stats["processed_frames"] == 5

    assert stats["total_detections"] == 10

    assert stats["class_counts"] == {
        "car": 5,
        "person": 5,
    }


def test_process_video_tracking(
    sample_video: Path,
    tmp_path: Path,
) -> None:
    """Test tracking path."""

    output_path = tmp_path / "tracked.mp4"

    stats = process_video(
        predictor=MockPredictor(),
        input_video_path=str(
            sample_video,
        ),
        output_video_path=str(
            output_path,
        ),
        tracking=True,
    )

    assert output_path.exists()

    assert stats["processed_frames"] == 5

    assert stats["total_detections"] == 10


def test_save_uploaded_video() -> None:
    """Test uploaded video save."""

    upload = MockUpload(
        b"video-bytes",
    )

    path = save_uploaded_video(
        upload,
    )

    assert Path(
        path,
    ).exists()
