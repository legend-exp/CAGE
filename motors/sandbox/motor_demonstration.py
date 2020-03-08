#!/usr/bin/env python3
import gclib
from pprint import pprint
import spur
import numpy as np
import time

def main():

    demonstration()

def demonstration():

    rotary_move = -90 #degrees
    linear_move = 20 #mm
    source_move = 45 #degrees

    print('Welcome to the CAGE motor demonstration program')
    time.sleep(3)
    print('The motors will move as follows:')
    time.sleep(1)
    print('Rotary stage: 90 degrees \nLinear stage: 2 cm \nSource motor: 45 degrees')
    time.sleep(1)
    print('Then each motor will move back to their respective positions')
    time.sleep(1)
    ans = input('Start movement? y/n \n -->')

    if ans == 'y':

        print('Starting demonstration...')
        time.sleep(2)

        rotary_program(rotary_move)
        linear_program(linear_move)
        source_program(source_move)

        print('Moving motors back to original position')

        rotary_program(-rotary_move)
        linear_program(-linear_move)
        source_program(-source_move)

    if ans == 'n':

        print('Fine, be that way')
        exit()


def rotary_program(angle):

    g = gclib.py()
    c = g.GCommand
    g.GOpen('172.25.100.168 --direct')

    zero = rotary_set_zero()
    while (zero > 10) and (zero < 16374):
        zero = rotary_set_zero()

    cts = angle * 12500

    if angle < 0:
        checks, rem = divmod(-cts, 25000)
        move = -25000
        rem = -1 * rem
    else:
        checks, rem = divmod(cts, 25000)
        move = 25000
    b = False
    i = 0

    c('AB')
    c('MO')
    c('SHD')
    c('SPD=15000')
    if load == 0:
        c('DPD=0')
    c('ACD=5000')
    c('BCD=5000')
    print(' Starting rotary movement...')

    if checks != 0:
        while i < checks:

            c('PRD={}'.format(move))
            c('BGD') #begin motion
            g.GMotionComplete('D')
            print(' encoder check')
            enc_pos = rotary_read_pos()

            if b == False:
                if (enc_pos > 8092) and (enc_pos < 8292):
                    print(' encoder position good, continuing')
                    theta = enc_pos * 360 / 2**14
                    print(theta, ' compared with 180')
                else:
                    print(' WARNING: Motor did not move designated counts, aborting move')
                    del c #delete the alias
                    g.GClose()
                    exit()
            if b == True:
                if (enc_pos < 100) or (enc_pos > 16284):
                    print(' encoder position good, continuing')
                    theta = enc_pos * 360 / 2**14
                    print(theta, ' compared with 0 or 360')
                else:
                    print(' WARNING1: Motor did not move designated counts, aborting move')
                    print((checks, rem, move, enc_pos, theta, b))
                    del c #delete the alias
                    g.GClose()
                    exit()
            b = not b
            i += 1

    if rem != 0:
        c('PRD={}'.format(rem))
        c('BGD') #begin motion
        g.GMotionComplete('D')

        print(' final encoder check', rem)
        enc_pos = rotary_read_pos()

        if rem < 0:
            bits = 2**14 + (rem * 2**14 / 50000)

            if b == False:
                if (enc_pos > (bits - 100)) and (enc_pos < (bits + 100)):
                    print(' encoder position good')
                    theta = enc_pos * 360 / 2**14
                    deg = rem * 360 / 50000
                    print(theta, ' compared with ', deg)
                else:
                    print(' WARNING: Motor did not move designated counts, aborting move')
                    del c #delete the alias
                    g.GClose()
                    exit()
            if b == True:
                if (enc_pos < (bits - 8092)) and (enc_pos > (bits - 8292)):
                     print(' encoder position good')
                     theta = enc_pos * 360 / 2**14
                     deg = rem * 360 / 50000 + 180
                     print(theta, ' compared with ', deg)
                else:
                    print(' WARNING2: Motor did not move designated counts, aborting move')
                    print((checks, rem, move, enc_pos, theta, b))
                    del c #delete the alias
                    g.GClose()
                    exit()

        else:
            bits = rem * 2**14 / 50000

            if b == False:
                if (enc_pos > (bits - 100)) and (enc_pos < (bits + 100)):
                    print(' encoder position good')
                    theta = enc_pos * 360 / 2**14
                    deg = rem * 360 / 50000
                    print(theta, ' compared with ', deg)
                else:
                    print(' WARNING: Motor did not move designated counts, aborting move')
                    del c #delete the alias
                    g.GClose()
                    exit()
            if b == True:
                if (enc_pos > (bits + 8092)) or (enc_pos < 100):
                     print(' encoder position good')
                     theta = enc_pos * 360 / 2**14
                     deg = rem * 360 / 50000 + 180
                     print(theta, ' compared with ', deg)
                else:
                    print(' WARNING: Motor did not move designated counts, aborting move')
                    del c #delete the alias
                    g.GClose()
                    exit()

    print(' Motor has moved to designated position')
    print('Motor counter: ', c('PAD=?'))
    del c #delete the alias
    g.GClose()

def linear_program(mm):

    g = gclib.py()
    c = g.GCommand
    g.GOpen('172.25.100.168 --direct')

    zero = linear_set_zero()
    while (zero > 10) and (zero < 16374):
        zero = linear_set_zero()

    cts = mm * 31573

    if mm < 0:
        checks, rem = divmod(-cts, 25000)
        move = -25000
        rem = -1 * rem
    else:
        checks, rem = divmod(cts, 25000)
        move = 25000
    b = False
    i = 0

    c('AB')
    c('MO')
    c('SHB')
    c('SPB=15000')
    if load == 0:
        c('DPB=0')
    c('ACB=5000')
    c('BCB=5000')
    print(' Starting linear move...')

    if checks != 0:
        while i < checks:

            c('PRB={}'.format(move))
            c('BGB') #begin motion
            g.GMotionComplete('B')
            print(' encoder check')
            enc_pos = linear_read_pos()

            if b == False:
                if (enc_pos > 8092) and (enc_pos < 8292):
                    print(' encoder position good, continuing')
                    theta = enc_pos * 360 / 2**14
                    print(theta, ' compared with 180')
                else:
                    print(' WARNING: Motor did not move designated counts, aborting move')
                    del c #delete the alias
                    g.GClose()
                    exit()
            if b == True:
                if (enc_pos < 100) or (enc_pos > 16284):
                    print(' encoder position good, continuing')
                    theta = enc_pos * 360 / 2**14
                    print(theta, ' compared with 0 or 360')
                else:
                    print(' WARNING: Motor did not move designated counts, aborting move')
                    del c #delete the alias
                    g.GClose()
                    exit()
            b = not b
            i += 1

    if rem != 0:
        c('PRB={}'.format(rem))
        c('BGB') #begin motion
        g.GMotionComplete('B')

        print(' encoder check')
        enc_pos = linear_read_pos()

        if rem < 0:
            bits = 2**14 + (rem * 2**14 / 50000)

            if b == False:
                if (enc_pos > (bits - 100)) and (enc_pos < (bits + 100)):
                    print(' encoder position good')
                    theta = enc_pos * 360 / 2**14
                    deg = rem * 360 / 50000
                    print(theta, ' compared with ', deg)
                else:
                    print(' WARNING: Motor did not move designated counts, aborting move')
                    del c #delete the alias
                    g.GClose()
                    exit()
            if b == True:
                if (enc_pos < (bits - 8092)) and (enc_pos > (bits - 8292)):
                     print(' encoder position good')
                     theta = enc_pos * 360 / 2**14
                     deg = rem * 360 / 50000 + 180
                     print(theta, ' compared with ', deg)
                else:
                    print(' WARNING: Motor did not move designated counts, aborting move')
                    del c #delete the alias
                    g.GClose()
                    exit()

        else:
            bits = rem * 2**14 / 50000

            if b == False:
                if (enc_pos > (bits - 100)) and (enc_pos < (bits + 100)):
                    print(' encoder position good')
                    theta = enc_pos * 360 / 2**14
                    deg = rem * 360 / 50000
                    print(theta, ' compared with ', deg)
                else:
                    print(' WARNING: Motor did not move designated counts, aborting move')
                    del c #delete the alias
                    g.GClose()
                    exit()
            if b == True:
                if (enc_pos > (bits + 8092)) or (enc_pos < 100):
                     print(' encoder position good')
                     theta = enc_pos * 360 / 2**14
                     deg = rem * 360 / 50000 + 180
                     print(theta, ' compared with ', deg)
                else:
                    print(' WARNING: Motor did not move designated counts, aborting move')
                    del c #delete the alias
                    g.GClose()
                    exit()

    print(' Motor has moved to designated position')
    print('Motor counter: ', c('PAB=?'))
    del c #delete the alias
    g.GClose()

def source_program():

    zero = source_set_zero()
    while (zero > 10) and (zero < 16374):
        zero = source_set_zero()

    cts = angle / 360 * 50000

    if angle < 0:
        checks, rem = divmod(-cts, 25000)
        move = -25000
        rem = -1 * rem
    else:
        checks, rem = divmod(cts, 25000)
        move = 25000
    b = False
    i = 0

    c('AB')
    c('MO')
    c('SHC')
    c('SPC=5000')
    if load == 0:
        c('DPC=0')
    c('ACC=5000')
    c('BCC=5000')
    print(' Starting move...')

    if checks != 0:
        while i < checks:

            c('PRC={}'.format(move))
            c('BGC') #begin motion
            g.GMotionComplete('C')
            print(' encoder check')
            enc_pos = source_read_pos()

            if b == False:
                if (enc_pos > 8092) and (enc_pos < 8292):
                    print(' encoder position good, continuing')
                    theta = enc_pos * 360 / 2**14
                    print(theta, ' compared with 180')
                else:
                    print(' WARNING: Motor did not move designated counts, aborting move')
                    del c #delete the alias
                    g.GClose()
                    exit()
            if b == True:
                if (enc_pos < 100) or (enc_pos > 16284):
                    print(' encoder position good, continuing')
                    theta = enc_pos * 360 / 2**14
                    print(theta, ' compared with 0 or 360')
                else:
                    print(' WARNING: Motor did not move designated counts, aborting move')
                    del c #delete the alias
                    g.GClose()
                    exit()
            b = not b
            i += 1

    if rem != 0:
        c('PRC={}'.format(rem))
        c('BGC') #begin motion
        g.GMotionComplete('C')

        print(' encoder check')
        enc_pos = source_read_pos()

        if rem < 0:
            bits = 2**14 + (rem * 2**14 / 50000)

            if b == False:
                if (enc_pos > (bits - 100)) and (enc_pos < (bits + 100)):
                    print(' encoder position good')
                    theta = enc_pos * 360 / 2**14
                    deg = rem * 360 / 50000
                    print(theta, ' compared with ', deg)
                else:
                    print(' WARNING: Motor did not move designated counts, aborting move')
                    del c #delete the alias
                    g.GClose()
                    exit()
            if b == True:
                if (enc_pos < (bits - 8092)) and (enc_pos > (bits - 8292)):
                     print(' encoder position good')
                     theta = enc_pos * 360 / 2**14
                     deg = rem * 360 / 50000 + 180
                     print(theta, ' compared with ', deg)
                else:
                    print(' WARNING: Motor did not move designated counts, aborting move')
                    del c #delete the alias
                    g.GClose()
                    exit()

        else:
            bits = rem * 2**14 / 50000

            if b == False:
                if (enc_pos > (bits - 100)) and (enc_pos < (bits + 100)):
                    print(' encoder position good')
                    theta = enc_pos * 360 / 2**14
                    deg = rem * 360 / 50000
                    print(theta, ' compared with ', deg)
                else:
                    print(' WARNING: Motor did not move designated counts, aborting move')
                    del c #delete the alias
                    g.GClose()
                    exit()
            if b == True:
                if (enc_pos > (bits + 8092)) or (enc_pos < 100):
                     print(' encoder position good')
                     theta = enc_pos * 360 / 2**14
                     deg = rem * 360 / 50000 + 180
                     print(theta, ' compared with ', deg)
                else:
                    print(' WARNING: Motor did not move designated counts, aborting move')
                    del c #delete the alias
                    g.GClose()
                    exit()

    print(' Motor has moved to designated position')
    print('Motor counter: ', c('PAC=?'))
    del c #delete the alias



if __name__=="__main__":
    main()
