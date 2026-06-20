"""
Tests for Arduino controller.
"""

from unittest.mock import MagicMock
from unittest.mock import patch

from src.arduino.controller import (
    ArduinoController,
)


@patch("src.arduino.controller.serial.Serial")
def test_connect(
    mock_serial,
) -> None:
    controller = ArduinoController(
        port="COM3",
    )

    controller.connect()

    mock_serial.assert_called_once()


@patch("src.arduino.controller.serial.Serial")
def test_disconnect(
    mock_serial,
) -> None:
    controller = ArduinoController(
        port="COM3",
    )

    controller.connect()

    serial_conn = controller.serial_connection

    controller.disconnect()

    serial_conn.close.assert_called_once()


@patch("src.arduino.controller.serial.Serial")
def test_send_message(
    mock_serial,
) -> None:
    controller = ArduinoController(
        port="COM3",
    )

    controller.connect()

    controller.send_message(
        "PED",
    )

    controller.serial_connection.write.assert_called_once()


@patch("src.arduino.controller.serial.Serial")
def test_send_event(
    mock_serial,
) -> None:
    controller = ArduinoController(
        port="COM3",
    )

    controller.connect()

    controller.send_event(
        "PEDESTRIAN_DETECTED",
    )

    controller.serial_connection.write.assert_called_once_with(
        b"PED\n",
    )


def test_unknown_event() -> None:
    controller = ArduinoController(
        port="COM3",
    )

    controller.serial_connection = MagicMock()

    controller.send_event(
        "UNKNOWN_EVENT",
    )

    controller.serial_connection.write.assert_not_called()
