"""
Arduino serial communication utilities.
"""

from __future__ import annotations

import time
import serial
import serial.tools.list_ports


class ArduinoController:
    """
    Pure serial transport layer for Arduino communication.
    """

    def __init__(
        self,
        port: str | None = None,
        baudrate: int = 9600,
        timeout: float = 1.0,
    ) -> None:

        self.port = port or self._auto_detect_port()
        self.baudrate = baudrate
        self.timeout = timeout

        self.serial_connection: serial.Serial | None = None

    # -------------------------
    # Auto detection
    # -------------------------
    def _auto_detect_port(self) -> str:
        ports = serial.tools.list_ports.comports()

        for p in ports:
            if "usbmodem" in p.device or "usbserial" in p.device:
                return p.device

        raise RuntimeError("Arduino not found. Check USB connection.")

    # -------------------------
    # Connection management
    # -------------------------
    def connect(self) -> None:
        if self.serial_connection is not None:
            return

        self.serial_connection = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=self.timeout,
        )

        time.sleep(2.0)  # Arduino reset delay

    def disconnect(self) -> None:
        if self.serial_connection is None:
            return

        self.serial_connection.close()
        self.serial_connection = None

    # -------------------------
    # Core I/O
    # -------------------------
    def send(self, command: str) -> None:
        if self.serial_connection is None:
            raise RuntimeError("Arduino is not connected.")

        try:
            payload = (command + "\n").encode("utf-8")
            self.serial_connection.write(payload)

        except serial.SerialException as e:
            self.serial_connection = None
            raise RuntimeError(f"Serial write failed: {e}")

    def read(self) -> str:
        if self.serial_connection is None:
            raise RuntimeError("Arduino is not connected.")

        return self.serial_connection.readline().decode().strip()

    # -------------------------
    # Context manager
    # -------------------------
    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect()
