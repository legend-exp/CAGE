#!/usr/bin/env python3
import gclib
from pprint import pprint
import spur
import numpy as np



def main():

    # test_readout()
    source_program()

def test_readout():

    g = gclib.py()
    c = g.GCommand
    g.GOpen('172.25.100.168 --direct')
    # print(type(g.GInfo()))
    # print(g.GAddresses())
    # motor_name = "DMC2142sH2a"

    # print(g.GVersion())
    # print(g.GInfo())


def source_program():

    g = gclib.py()
    c = g.GCommand
    g.GOpen('172.25.100.168 --direct')

    zero = set_zero()
    while (zero > 10) and (zero < 16374):
        zero = set_zero()

    load = int(input(' If you are starting a move, type 0. \n If you are moving back to 0 position, type 1 \n -->'))

    if load == 0:
        mm = float(input(' How many mm would you like to move the linear motor?\n -->'))
        pos = np.asarray([mm])
        np.savez('linear_pos', pos)
    if load == 1:
        print(' Setting motor back to 0 position')
        file = np.load('./linear_pos.npz')
        mm1 = file['arr_0']
        mm = -mm1[0]
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
    # print(checks, rem)
    # del c #delete the alias
    # g.GClose()
    # exit()


    c('AB')
    c('MO')
    c('SHB')
    c('SPB=15000')
    if load == 0:
        c('DPB=0')
    c('ACB=5000')
    c('BCB=5000')
    print(' Starting move...')

    if checks != 0:
        while i < checks:

            c('PRB={}'.format(move))
            c('BGB') #begin motion
            g.GMotionComplete('B')
            print(' encoder check')
            enc_pos = read_pos()

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
        enc_pos = read_pos()

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

def read_pos():

    shell = spur.SshShell(hostname="10.66.193.75",
                            username="pi", password="raspberry")

    with shell:
        result = shell.run(["python3", "read_pos_linear.py"])
    answer = result.output
    ans = float(answer.decode("utf-8"))
    print("Real position is: ", ans)
    return ans

def set_zero():

    shell = spur.SshShell(hostname="10.66.193.75",
                            username="pi", password="raspberry")

    with shell:
        result = shell.run(["python3", "set_zero_linear.py"])
    # answer = result.output
    ans = float(result.output.decode("utf-8"))
    print("Encoder set to zero, returned: ", ans)
    return ans

if __name__=="__main__":
    main()
