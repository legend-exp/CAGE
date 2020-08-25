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
t_sleep = 0.1

pos = [0x00, 0x00]
reset = [0x00, 0x60]
zero = [0x00, 0x70]

# communicate with the encoder
GPIO.output(7, GPIO.LOW)
time.sleep(t_sleep)
val = spi.xfer2(pos)
time.sleep(t_sleep)
GPIO.output(7, GPIO.HIGH)
if len(val) != 2:
    print("ERROR")
    sleep(1)

# bitshift to get answer -- why don't we return or print this?
reply = val[0]<<8
reply |= val[1]
time.sleep(t_sleep)

# zero the encoder
GPIO.output(7, GPIO.LOW)
time.sleep(t_sleep)
# val = spi.xfer2(zero) # this one is iffy
val = spi.xfer2(reset) # seems like this one works better
time.sleep(t_sleep)
GPIO.output(7, GPIO.HIGH)
if len(val) != 2:
    print("ERROR")
    sleep(1)

# bitshift to get answer
reply = val[0]<<8
reply |= val[1]

# this is our result.
print(reply & 0x3FFF)

time.sleep(t_sleep)
GPIO.cleanup()
spi.close()
exit()
