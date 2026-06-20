"""
Detection event generation (stateful + debounced).
"""

from __future__ import annotations


class EventGenerator:
    """
    Converts raw class counts into stable high-level events.

    Uses state tracking to avoid repeated event spam
    across video frames (important for YOLO pipelines).
    """

    def __init__(self) -> None:
        self._state = {
            "person_present": False,
            "heavy_traffic": False,
            "traffic_signal_visible": False,
        }

    def generate(
        self,
        class_counts: dict[str, int],
    ) -> list[str]:
        events: list[str] = []

        vehicle_count = (
            class_counts.get("car", 0)
            + class_counts.get("truck", 0)
            + class_counts.get("bus", 0)
        )

        # -------------------------
        # Pedestrian event (edge-triggered)
        # -------------------------
        person_present = class_counts.get("person", 0) > 0

        if person_present and not self._state["person_present"]:
            events.append("PEDESTRIAN_DETECTED")

        self._state["person_present"] = person_present

        # -------------------------
        # Heavy traffic event (edge-triggered threshold crossing)
        # -------------------------
        heavy_traffic = vehicle_count > 20

        if heavy_traffic and not self._state["heavy_traffic"]:
            events.append("HEAVY_TRAFFIC")

        self._state["heavy_traffic"] = heavy_traffic

        # -------------------------
        # Traffic signal visibility (edge-triggered)
        # -------------------------
        traffic_light = class_counts.get("traffic light", 0) > 0

        if traffic_light and not self._state["traffic_signal_visible"]:
            events.append("TRAFFIC_SIGNAL_VISIBLE")

        self._state["traffic_signal_visible"] = traffic_light

        return events
