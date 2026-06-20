"""
Event → Command mapping layer.
Pure deterministic translation logic.
"""

from __future__ import annotations

EVENT_TO_COMMAND: dict[str, str] = {
    "PEDESTRIAN_DETECTED": "PED",
    "HEAVY_TRAFFIC": "TRAFFIC",
    "TRAFFIC_SIGNAL_VISIBLE": "LIGHT",
}


def map_event(event: str) -> str | None:
    """
    Convert high-level event into Arduino command.

    Args:
        event:
            Semantic event name.

    Returns:
        Arduino command string or None if unmapped.
    """
    return EVENT_TO_COMMAND.get(event)
