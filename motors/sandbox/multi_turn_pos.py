import serial
from serial.tools.list_ports import comports

[x.device for x in comports()]

serial = serial.Serial('/dev/ttyUSB0')
