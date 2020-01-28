#!/usr/bin/env python3
import gclib
from pprint import pprint
import spur
import numpy as np



def main():

    # test_readout()
    test_program()

def test_readout():

    g = gclib.py()
    c = g.GCommand
    g.GOpen('172.25.100.168 --direct')
    # print(type(g.GInfo()))
    # print(g.GAddresses())
    # motor_name = "DMC2142sH2a"

    # print(g.GVersion())
    # print(g.GInfo())


def test_program():

    g = gclib.py()
    c = g.GCommand
    g.GOpen('172.25.100.168 --direct')

    zero = set_zero()
    while (zero > 10) and (zero < 16374):
        zero = set_zero()

    angle = float(input(' How many degrees would you like to rotate source motor?\n -->'))
    cts = angle / 360 * 50000
    ratio1 = cts / 50000

    c('AB')
    c('MO')
    c('SHC')
    c('SPC=5000')
    c('DPC=0')
    c('PRC={}'.format(cts))
    c('ACC=5000')
    c('BCC=5000')
    print(' Starting move...')
    c('BGC') #begin motion
    g.GMotionComplete('C')
    print(' done.')

    print(' Checking position...')
    enc_pos = read_pos()
    ratio2 = enc_pos / (2**14)
    print(ratio1, '\n', ratio2)

    if (ratio2 < .95 * ratio1) or (ratio2 > 1.05 * ratio1):
        print(' WARNING: Motor did not move designated counts')
    else:
        print('Motor moved designated counts')

    del c #delete the alias
    g.GClose()

def read_pos():

    shell = spur.SshShell(hostname="10.66.193.74",
                            username="pi", password="raspberry")

    with shell:
        result = shell.run(["python3", "pos.py"])
    answer = result.output
    ans = float(answer.decode("utf-8"))
    print("Real position is: ", ans)
    return ans

def set_zero():

    shell = spur.SshShell(hostname="10.66.193.74",
                            username="pi", password="raspberry")

    with shell:
        result = shell.run(["python3", "set_zero.py"])
    # answer = result.output
    ans = float(result.output.decode("utf-8"))
    print("Encoder set to zero, returned: ", ans)
    return ans



if __name__=="__main__":
    main()
