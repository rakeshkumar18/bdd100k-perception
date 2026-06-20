import serial
import serial.tools.list_ports
import time


def find_arduino_port():
    ports = serial.tools.list_ports.comports()
    for p in ports:
        if "usbmodem" in p.device or "usbserial" in p.device:
            return p.device
    return None


port = find_arduino_port()

if not port:
    raise Exception("Arduino not found. Check USB connection.")

print("Using port:", port)

arduino = serial.Serial(port, 9600, timeout=1)
time.sleep(2)

commands = ["LED_ON\n", "LED_OFF\n", "MOTOR:120\n"]

for cmd in commands:
    arduino.write(cmd.encode())
    print("Sent:", cmd.strip())
    time.sleep(2)

arduino.close()
