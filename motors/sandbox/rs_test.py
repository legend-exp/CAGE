import time
import serial

ser = serial.Serial('/dev/ttyUSB0', baudrate=9600,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    bytesize=serial.EIGHTBITS
                    )

time.sleep(1)
try:
    ser.write('Hello World')
    print('Data worked')
    while True:
        if ser.inWaiting() > 0:
            data = ser.read()
            print(data)
except KeyboardInterrupt:
    print('Exiting Program')
except:
    print('Errors occured')
finally:
    ser.close()
    pass
