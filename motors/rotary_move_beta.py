#!/usr/bin/env python3
import gclib
from pprint import pprint
import spur
import numpy as np

def main():
    """
    NOTE: rotary motor is Axis D in GalilTools.
    """
    # test_readout()
    # zero_rotary_motor()
    rotary_program()


def test_readout():

    g = gclib.py()
    c = g.GCommand
    g.GOpen('172.25.100.168 --direct')
    print(type(g.GInfo()))
    print(g.GAddresses())
    motor_name = "DMC2142sH2a"
    print(g.GVersion())
    print(g.GInfo())


def rotary_limit_check():
    """
    rotary motor only has one limit switch
    """
    g = gclib.py()
    g.GOpen('172.25.100.168 --direct')

    # send the command (can check against GalilTools)
    status = int(float(g.GCommand('MG _LF D')))
    # status = int(g.GCommand('MG _LR D')) # reverse isn't used

    label = "OFF" if status == 1 else "ON"
    print(f"Rotary motor limit switch: {label}")


def rotary_program():
    """
    NOTE: the values the encoder is returning are pretty wack.
    Especially if you run the loop "read_rotary.py" on the pi concurrently.
    We need much more careful handling of the values the encoder reads
    back (assuming there isn't a hardware issue).  I think it's solvable,
    but it requires attention to get this right.
    """

    g = gclib.py()
    c = g.GCommand
    g.GOpen('172.25.100.168 --direct')

    # load = int(input(' If you are starting a move, type 0. \n If you are moving back to 0 position, type 1(test, do not use) \n -->'))

    # set the current encoder position to zero (measures relative motion)
    zero = rotary_set_zero()
    while (zero > 10) and (zero < 16374):
        zero = rotary_set_zero()

    # if load == 0:
    #     angle = float(input(' How many degrees would you like to rotate the rotary motor?\n NOTE:negative angles rotate away from limit switch \n -->'))
    #     pos = np.asarray([angle])
    #     np.savez('rotary_pos', pos)
    #
    # if load == 1:
    #     print(' Setting motor back to 0 position')
    #     file = np.load('./rotary_pos.npz')
    #     angle1 = file['arr_0']
    #     angle = -angle1[0]

    # debug, deleteme!
    load = 0
    # angle = -45 # doesn't work, can't make it through the while loop
    angle = 1

    if abs(angle) > 330:
        print('Too great of an angle, no angles greater than 330.')
        exit()

    # convert degrees into number of steps
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

    # send commands to motor
    c('AB')
    c('MO')
    c('SHD')
    c('SPD=300000')
    if load == 0:
        c('DPD=0')
    c('ACD=300000')
    c('DCD=300000')
    print(' Starting move...')

    if checks != 0:
        while i < checks:

            print(f"i {i}  n_checks {checks}  cts {cts}  move {move}")

            c('PRD={}'.format(move))
            c('BGD') #begin motion
            g.GMotionComplete('D')
            enc_pos = rotary_read_pos()
            print(f' encoder check: {enc_pos}')

            if b == False:
                if (enc_pos > 8092) and (enc_pos < 8292):
                    print(' encoder position good, continuing')
                    theta = enc_pos * 360 / 2**14
                    print(theta, ' compared with 180')
                else:
                    print(' WARNING: Motor did not move designated counts, aborting move')
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

    g.GClose()


def zero_rotary_motor():

    g = gclib.py()
    c = g.GCommand
    g.GOpen('172.25.100.168 --direct')

    print(' Attempting to zero the rotary motor now, sudden error or break in code expected')
    print(' Rerun motor_movement.py to continue')

    # zero = rotary_set_zero()
    # while (zero > 10) and (zero < 16374):
    #     zero = rotary_set_zero()

    b = False
    move = 25000

    c('AB')
    c('MO')
    c('SHD')
    c('SPD=50000')
    c('ACD=300000')
    c('BCD=300000')
    print(' Starting move...')

    try:
        while True:

            c('PRD={}'.format(move))
            c('BGD') #begin motion
            g.GMotionComplete('D')
            enc_pos = rotary_read_pos()

            print(f"encoder position: {enc_pos}")


            if b == False:
                if (enc_pos > 8092) and (enc_pos < 8292):
                    print(' encoder position good, continuing')
                    theta = enc_pos * 360 / 2**14
                    print(theta, ' compared with 180')
                else:
                    print(' WARNING: Motor did not move designated counts, aborting move')
                    g.GClose()
                    exit()

            if b == True:
                if (enc_pos < 100) or (enc_pos > 16284):
                    print(' encoder position good, continuing')
                    theta = enc_pos * 360 / 2**14
                    print(theta, ' compared with 0 or 360')
                else:
                    print(' WARNING1: Motor did not move designated counts, aborting move')
                    g.GClose()
                    exit()
            b = not b

    except gclib.GclibError:
            print('Rotary stage is zeroed')

    g.GClose()


def rotary_read_pos():

    shell = spur.SshShell(hostname="10.66.193.75",
                          username="pi", password="raspberry")

    with shell:
        # this has type: spur.results.ExecutionResult
        result = shell.run(["python3", "encoders/pos_rotary.py"])

    answer = result.output
    ans = float(answer.decode("utf-8"))
    print("Real position is: ", ans)
    return ans


def rotary_set_zero():

    shell = spur.SshShell(hostname="10.66.193.75",
                          username="pi", password="raspberry")

    with shell:
        result = shell.run(["python3", "encoders/pos_rotary.py"])

    ans = float(result.output.decode("utf-8"))
    print("Encoder set to zero, returned: ", ans)
    return ans


if __name__=="__main__":
    main()
