from src.arduino.controller import ArduinoController
import time

with ArduinoController() as arduino:

    arduino.send("PED")
    print("LED ON")

    time.sleep(5)

    arduino.send("OFF")
    print("LED OFF")