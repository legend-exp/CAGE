import RPi.GPIO as GPIO
import spidev
import time

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
GPIO.setup(7, GPIO.OUT)
GPIO.output(7, GPIO.HIGH)

spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 200000

pos = [0x00, 0x00]
reset = [0x00, 0x60]
zero = [0x00, 0x70]
while True:

    GPIO.output(7, GPIO.LOW)
    time.sleep(0.075)

    val = spi.xfer2(pos)

    time.sleep(0.075)
    GPIO.output(7, GPIO.HIGH)

    if len(val) != 2:
        print("ERROR, rerun script")
        sleep(1)
    rep1 = val[0]<<8
    rep1 |= val[1]
    print(rep1 & 0x3FFF)
    time.sleep(1)

GPIO.cleanup()
spi.close()
exit()
