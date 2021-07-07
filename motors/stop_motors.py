#!/usr/bin/env python3
import gclib
import json
import time

gp = gclib.py()
gc = gp.GCommand
with open('../config.json') as f:
    ipconf = json.load(f)
with open('motor_config.json') as g:
    motorDB = json.load(g)
mconf = motorDB["mconf"]


# connect to controller
mip = ipconf["newmark"]
try:
    gp.GOpen(f"{mip} --direct")
except:
    print("ERROR: couldn't connect to Newmark controller!")

    # try pinging it before you give up
    ping = os.system("ping -c 1 " + mip)
    if ping == 0:
        print("Controller is online ...")
    else:
        print("Controller isn't active on the network!  Beware ...")

# if verbose:
# print("\nConnected to Newmark controller.")
# print(f"  {gp.GInfo()}")
# print(f"  {gp.GVersion()}")

# declare which motor to move and how far
# user, you can change this: 'source', 'linear', or 'rotary'
motor_name = 'source'

axis = mconf[motor_name]['axis']
mspd = mconf[motor_name]['motor_spd']
print(axis, mspd)

# n_steps = -25000 # default: move -180 degrees
# n_steps = 25000 # interesting: limit switch will stop this + motion but not the - one

# initial motor setup commands
# gc('AB')
# gc('MO')
# gc(f'SH{axis}')
# gc(f'SP{axis}={mspd}')
# gc(f'DP{axis}=0')
# gc(f'AC{axis}={mspd}')
# gc(f'BC{axis}={mspd}')

res = gc(f'ST{axis}')
success = res is not None
print('Did we succeed at stopping the motion?', success)
move_completed = False
gp.GMotionComplete(axis)
