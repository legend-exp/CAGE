import RPi.GPIO as GPIO
import spidev
import time

# originally David P's code.  Modified to work w/ source encoder.

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
GPIO.setup(13, GPIO.OUT) # this is RPi pin (also "GPIO 27")
GPIO.output(13, GPIO.HIGH)

spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 10000 # note: this is david's setting

pos = [0x00, 0x00]
reset = [0x00, 0x60]
zero = [0x00, 0x70]

while(1):
    GPIO.output(13, GPIO.LOW)
    time.sleep(0.001)
    #pos = [10,10]
    #print(pos)
    val = spi.xfer2(pos)
    pos=[0,0]
    #print(pos)
    time.sleep(0.02)
    GPIO.output(13, GPIO.HIGH)

    if len(val) != 2:
        print("ERROR, rerun script")
        sleep(1)

    rep1 = val[0]<<8
    rep1 |= val[1]
    print(rep1 & 0x3FFF)
    time.sleep(.1)

GPIO.cleanup()
spi.close()

exit()
