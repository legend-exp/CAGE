#!/usr/bin/env python3
import gclib
from pprint import pprint
import spur
import numpy as np
import json

def main():
    # test_readout()
    source_limit_check()
    # source_program()


def test_readout():

    g = gclib.py()
    g.GOpen('172.25.100.168 --direct')

    print(type(g.GInfo()))
    print(g.GAddresses())
    motor_name = "DMC2142sH2a"
    print(g.GVersion())
    print(g.GInfo())


def source_limit_check():
    """
    source motor only has one limit switch
    """
    g = gclib.py()
    g.GOpen('172.25.100.168 --direct')

    # send the command (can check against GalilTools)
    status = int(float(g.GCommand('MG _LF A')))
    # status = int(g.GCommand('MG _LR A')) # reverse isn't used

    label = "OFF" if status == 1 else "ON"
    print(f"Source motor limit switch: {label}")


def source_program():

    g = gclib.py()
    c = g.GCommand
    g.GOpen('172.25.100.168 --direct')

    zero = source_set_zero()
    while (zero > 10) and (zero < 16374):
        zero = source_set_zero()

    load = int(input(' If you are starting a move, type 0. \n If you are moving back to 0 position, type 1 \n -->'))

    if load == 0:
        angle = float(input(' How many degrees would you like to rotate source motor?\n Note:negative angles move motor away from limit switch \n  -->'))
        pos = np.asarray([angle])
        np.savez('source_pos', pos)
    if load == 1:
        print(' Setting motor back to 0 position')
        file = np.load('./source_pos.npz')
        angle1 = file['arr_0']
        angle = -angle1[0]
    cts = angle / 360 * 50000

    if angle < -25:
        print('WARNING, cannot rotate source farther than 25 degrees past normal incidence, away from limit switch.')
        print('Restart motor movement program and choose smaller angle')
        exit()


    if angle < 0:
        checks, rem = divmod(-cts, 25000)
        move = -25000
        rem = -1 * rem
    else:
        checks, rem = divmod(cts, 25000)
        move = 25000
    b = False
    i = 0
    # print(checks, rem)
    # del c #delete the alias
    # g.GClose()
    # exit()


    c('AB')
    c('MO')
    c('SHA')
    c('SPA=5000')
    if load == 0:
        c('DPC=0')
    c('ACA=5000')
    c('BCA=5000')
    print(' Starting move...')

    if checks != 0:
        while i < checks:

            c('PRA={}'.format(move))
            c('BGA') #begin motion
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
        c('PRA={}'.format(rem))
        c('BGA') #begin motion
        g.GMotionComplete('A')

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
    print('Motor counter: ', c('PAA=?'))
    del c #delete the alias
    g.GClose()


def source_set_zero():

    # load credentials
    with open("../config.json") as f:
        config = json.load(f)
    ipa = config["cage_rpi_ipa"]
    usr = config["cage_rpi_usr"]
    pwd = config["cage_rpi_pwd"]

    shell = spur.SshShell(hostname=ipa, username=usr, password=pwd)
    with shell:
        result = shell.run(["python3", "encoders/set_zero_source.py"])

    ans = float(result.output.decode("utf-8"))
    print("Encoder set to zero, returned: ", ans)
    return ans


def zero_source_motor():

    g = gclib.py()
    c = g.GCommand
    g.GOpen('172.25.100.168 --direct')

    zero = source_set_zero()
    while (zero > 10) and (zero < 16374):
        zero = source_set_zero()

    print(' Attempting to zero the source motor now, sudden error or break in code expected')
    print(' Rerun motor_movement.py to continue')

    b = False
    move = 25000

    c('AB')
    c('MO')
    c('SHA')
    c('SPA=5000')
    c('ACA=5000')
    c('BCA=5000')
    print(' Starting move...')

    try:
        while True:

            c('PRA={}'.format(move))
            c('BGA') #begin motion
            g.GMotionComplete('A')
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

    except gclib.GclibError:
            print('Source Motor is zeroed')

    del c #delete the alias
    g.GClose()


def center_source_motor():

    zero = source_set_zero()
    while (zero > 10) and (zero < 16374):
        zero = source_set_zero()

    print(' Centering source beam, normal to the detector surface.')

    g = gclib.py()
    c = g.GCommand
    g.GOpen('172.25.100.168 --direct')

    load = 0
    b = False
    angle = -180
    cts = -25502


    if angle < 0:
        checks, rem = divmod(-cts, 25000)
        move = -25000
        rem = -1 * rem
    else:
        checks, rem = divmod(cts, 25000)
        move = 25000
    b = False
    i = 0
    # print(checks, rem)
    # del c #delete the alias
    # g.GClose()
    # exit()


    c('AB')
    c('MO')
    c('SHA')
    c('SPA=5000')
    if load == 0:
        c('DPC=0')
    c('ACA=5000')
    c('BCA=5000')
    print(' Starting move...')

    if checks != 0:
        while i < checks:

            c('PRA={}'.format(move))
            c('BGA') #begin motion
            g.GMotionComplete('A')
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
        c('PRA={}'.format(rem))
        c('BGA') #begin motion
        g.GMotionComplete('A')

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
    print('Motor counter: ', c('PAA=?'))
    del c #delete the alias
    g.GClose()


def source_read_pos():

    shell = spur.SshShell(hostname="10.66.193.75",
                            username="pi", password="raspberry")

    with shell:
        result = shell.run(["python3", "pos.py"])
    answer = result.output
    ans = float(answer.decode("utf-8"))
    print("Real position is: ", ans)
    return ans


if __name__=="__main__":
    main()
