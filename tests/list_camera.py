import cv2
import time

from src.dashboard.components.inference_utils import (
    load_predictor,
)

predictor = load_predictor()

frame = cv2.imread(
    "/Users/rk/Downloads/100k/test/cabc30fc-eb673c5a.jpg"
)

times = []

for _ in range(20):

    start = time.perf_counter()

    predictor.predict_frame(
        frame=frame,
        conf=0.25,
    )

    times.append(
        time.perf_counter() - start
    )

avg = sum(times) / len(times)

print(f"Average: {avg:.3f}s")
print(f"FPS: {1/avg:.2f}")