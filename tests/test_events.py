from src.inference.events import (
    generate_events,
)


def test_pedestrian_event():
    events = generate_events(
        {
            "person": 5,
        }
    )

    assert "PEDESTRIAN_DETECTED" in events


def test_heavy_traffic_event():
    events = generate_events(
        {
            "car": 15,
            "truck": 5,
            "bus": 3,
        }
    )

    assert "HEAVY_TRAFFIC" in events


def test_traffic_signal_event():
    events = generate_events(
        {
            "traffic light": 1,
        }
    )

    assert "TRAFFIC_SIGNAL_VISIBLE" in events


def test_no_events():
    events = generate_events({})

    assert events == []
