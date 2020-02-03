import spidev
import time
import RPi.GPIO as GPIO

# pi11 -- checks position of rotary motor
# odd -- sometimes this needs to be run twice?

spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 10000
t_sleep = 0.1

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
GPIO.setup(11, GPIO.OUT)
GPIO.output(11, GPIO.HIGH)

pos = [0x00,0x00]
reset = [0x00,0x60]
zero = [0x00,0x70]

GPIO.output(11, GPIO.LOW)
time.sleep(t_sleep)
val = spi.xfer2(pos)
time.sleep(t_sleep)
GPIO.output(11, GPIO.HIGH)

if len(val) != 2:
    print("ERROR, rerun script")
    sleep(t_sleep)

# bitshift the result
rep1 = val[0]<<8
rep1 |= val[1]
print(rep1 & 0x3FFF)

time.sleep(t_sleep)

GPIO.cleanup()
spi.close()
exit()
