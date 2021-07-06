#!/usr/bin/env python3
import multiprocessing
import time
import gclib
import json
import sys

def main():

    # set parameters
    with open('../config.json') as f:
        ipconf = json.load(f)
    with open('motor_config.json') as g:
        motorDB = json.load(g)
    mconf = motorDB["mconf"]

    motor_name = 'source'
    axis = mconf[motor_name]['axis']
    mspd = mconf[motor_name]['motor_spd']
    mip = ipconf['newmark']

    # n_steps = -25000 # default: move -180 degrees (AWAY from lim switch)
    # n_steps = 25000 # interesting: limit switch will stop this + motion but not the - one.  this is hardwired

    n_steps = -25000/2 # move AWAY from lim switch 90 deg
    # n_steps = 25000/2  # move BACK to lim switch 90 deg
    # n_steps = -1000



    pool = multiprocessing.Pool(1)
    try:
        pool.apply_async(run_motion, args=(mip, axis, mspd, n_steps), callback=cb)
        pool.close()
        pool.join()

    except KeyboardInterrupt:
        print('\nGot emergency ctrl-c stop signal!')
        pool2 = multiprocessing.Pool(1)
        pool2.apply_async(stop_motion, args=(mip, axis), callback=cb)
        pool.terminate()
        pool2.close()
        pool2.join()
        print("WARNING: You've used emergency stop.  The motor may still be\n",
              "   'humming', i.e. powered but not moving.  Sometimes this is\n",
              "   audible, but not always. You can either try \n",
              "   another (safe) motion to reset it, or reset the Newmark\n",
              "   motor controller.")

    print('done')


def cb(this):
    print('callback:', this)


def run_motion(mip, axis, mspd, n_steps):
    # must be global for multiprocessing to work.

    # connect to controller
    gp = gclib.py()
    gc = gp.GCommand
    try:
        gp.GOpen(f"{mip} --direct")
    except:
        print("ERROR: couldn't connect to Newmark controller!")

    # initial motor setup commands
    gc('AB')
    gc('MO')
    gc(f'SH{axis}')
    gc(f'SP{axis}={mspd}')
    gc(f'DP{axis}=0')
    gc(f'AC{axis}={mspd}')
    gc(f'BC{axis}={mspd}')

    # begin motion
    gc(f'PR{axis}={n_steps}')
    gc(f'BG{axis}')
    gp.GMotionComplete(axis)
    time.sleep(.1)

    print('i did a thing')


def stop_motion(mip, axis):

    # connect to controller
    gp = gclib.py()
    gc = gp.GCommand
    try:
        gp.GOpen(f"{mip} --direct")
    except:
        print("ERROR: couldn't connect to Newmark controller!")

    # -- emergency stop signal --
    res = gc(f'ST{axis}')
    success = res is not None
    print('Did we succeed at stopping the motion?', success)
    gp.GMotionComplete(axis)
    time.sleep(.1)




if __name__ == "__main__":
    main()
