#!/usr/bin/env python3
import argparse
import time
import spidev
import RPi.GPIO as GPIO


def main():
    """ 
    CAGE encoder readout utility.
    This is run locally on the CAGE RPi, typically via SSH.
    
    Access cmd-line help with:
        $ python3 read_encoders.py -h
        
    NOTES: 
    * CAGE RPi currently runs Python 3.5.3, so no f-strings are allowed.
    * trick to quickly sync, assuming you're in the cage/motors directory:
      (it helps to have SSH keys set up)
        $ rsync -av ./ pi@10.66.193.75:/home/pi/cage/motors
    """
    par = argparse.ArgumentParser(description='CAGE encoder readout utility')
    arg, st, sf = par.add_argument, 'store_true', 'store_false'
    
    # encoder functions
    arg('-p', '--pos', nargs=1, type=int, help='single-read encoder on RPi pin')
    arg('-l', '--loop', nargs=1, type=int, help='continuous-read on RPi pin')
    arg('-z', '--zero', nargs=1, type=int, help='set encoder value to zero')
    
    # encoder settings (override defaults)
    arg('-s', '--sleep', nargs=1, type=float, help='set sleep time')
    arg('-c', '--comm', nargs=1, type=int, help='set RPi comm speed (Hz)')
    arg('-v', '--verbose', action=st, help='set verbose output')
    arg('-m', '--n_max', nargs=1, type=int, help='set num. tries to zero encoder')

    args = vars(par.parse_args()) # convert to dict
    
    # modify default settings
    t_sleep = args['sleep'][0] if args['sleep'] else 0.01 # sec
    com_spd = args['comm'][0] if args['comm'] else 10000  # Hz
    verbose = args['verbose']
    n_max = args['n_max'][0] if args['n_max'] else 3 # tries
    
    # handy globals (used by almost all routines)
    global enc_address
    enc_address = {
        'pos': [0x00, 0x00],
        'reset': [0x00, 0x60],
        'zero': [0x00, 0x70]
        }
    
    # run routines
    if args['pos']:
        rpi_pin = int(args['pos'][0])
        read_pos(rpi_pin, t_sleep, com_spd, verbose)
        
    if args['loop']:
        rpi_pin = int(args['loop'][0])
        verbose = True # override user arg, we need to see output always
        while True:
            read_pos(rpi_pin, t_sleep, com_spd, verbose)
        
    if args['zero']:
        rpi_pin = int(args['zero'][0])
        set_zero(rpi_pin, n_max, t_sleep, com_spd, verbose)


def read_pos(rpi_pin, t_sleep=0.01, com_spd=10000, verbose=True):
    """
    read the current position of an encoder, a single time
    """
    global enc_address

    # open spi/rpi connection
    spi = spidev.SpiDev()
    spi.open(0, 0)
    spi.max_speed_hz = com_spd
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)
    GPIO.setup(rpi_pin, GPIO.OUT)
    GPIO.output(rpi_pin, GPIO.HIGH)

    # open communication and read position
    GPIO.output(rpi_pin, GPIO.LOW)
    time.sleep(t_sleep)
    enc_val = spi.xfer2(enc_address['pos'])
    time.sleep(t_sleep)
    GPIO.output(rpi_pin, GPIO.HIGH)
    time.sleep(t_sleep)

    if len(enc_val) != 2:
        print('ERROR: encoder value is nonstandard:', enc_val)
        exit()

    # bitshift the result
    rep1 = enc_val[0] << 8
    rep1 |= enc_val[1]
    result = rep1 & 0x3FFF
    
    if verbose:
        print(result)

    # cleanup and return
    GPIO.cleanup()
    spi.close()
    
    return result


def set_zero(rpi_pin, n_max=3, t_sleep=0.01, com_spd=10000, verbose=True):
    """
    read the encoder's current value, and then send the 'reset' command.
    this allows us to more accurately track the current encoder position, 
    instead of allowing the value to cycle past 2**14=16384 haphazardly.
    
    The quirks of the encoder require we first read the current position,
    then send the reset command, so we can't call read_pos here.
    
    NOTE: Tim found that sometimes this has to be run twice.  So I put in a
    while loop here that tries `n_max` times to zero the encoder, before
    giving up.
    """
    global enc_address

    # open spi/rpi connection
    spi = spidev.SpiDev()
    spi.open(0, 0)
    spi.max_speed_hz = com_spd
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)
    GPIO.setup(rpi_pin, GPIO.OUT)
    GPIO.output(rpi_pin, GPIO.HIGH)
    
    # check if zeroed
    zeroed = False
    counter = 0
    while zeroed is not True and counter <= n_max:
        
        # open communication and read position
        GPIO.output(rpi_pin, GPIO.LOW)
        time.sleep(t_sleep)
        enc_val = spi.xfer2(enc_address['pos'])
        time.sleep(t_sleep)
        GPIO.output(rpi_pin, GPIO.HIGH)
        time.sleep(t_sleep)
        if len(enc_val) != 2:
            print('ERROR: encoder value is nonstandard:', enc_val)
            exit()
        rep1 = enc_val[0] << 8
        rep1 |= enc_val[1]
        start_pos = rep1 & 0x3FFF    
        time.sleep(t_sleep)
        
        # send command to zero the encoder
        GPIO.output(rpi_pin, GPIO.LOW)
        # time.sleep(t_sleep)
        # enc_val = spi.xfer2(enc_address['reset'])
        time.sleep(t_sleep)
        enc_val = spi.xfer2(enc_address['zero']) # tim says this should work
        time.sleep(t_sleep)
        GPIO.output(rpi_pin, GPIO.HIGH)
        if len(enc_val) != 2:
            print('ERROR: encoder value is nonstandard:', enc_val)
            exit()
        rep1 = enc_val[0] << 8
        rep1 |= enc_val[1]
        zeroed_pos = rep1 & 0x3FFF 

        # OK to be within +/- 10 steps of 0
        zeroed = True if zeroed_pos < 10 or zeroed_pos > 16374 else False
        counter += 1
        
        if verbose:
            print("i {}  start {}  end {}  zeroed? {}".format(counter, start_pos, zeroed_pos, zeroed))
        
    if not zeroed and counter == n_max:
        print("ERROR, couldn't zero the encoder. UGGG. Final pos:", zeroed_pos)
        
    # these values are read by motor_movement
    print(start_pos, zeroed_pos, zeroed)
    
    # cleanup and return
    GPIO.cleanup()
    spi.close()
    
    return start_pos, zeroed_pos, zeroed
    
    
    
    
if __name__=='__main__':
    main()
    