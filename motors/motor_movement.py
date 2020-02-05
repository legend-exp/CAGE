#!/usr/bin/env python3
import os
import time
import json
import shlex
import argparse
import gclib
import spur
import numpy as np
from pprint import pprint

def main():
    doc="""
    CAGE motor movement suite.
    
    REFERENCES:
    https://elog.legend-exp.org/UWScanner/200130_140616/cage_electronics.pdf
    http://www.galilmc.com/sw/pub/all/doc/gclib/html/python.html
    """
    # load configuration (uses globals, it's bad practice but who cares)
    global gp, gc, conf, ipconf, mconf, rpins
    gp = gclib.py()
    gc = gp.GCommand
    with open('../config.json') as f:
        ipconf = json.load(f)
    mconf = {
        'source': {'rpi_pin':13, 'axis':'A', 'motor_spd':5000},
        'linear': {'rpi_pin':7, 'axis':'B', 'motor_spd':300000},
        'rotary': {'rpi_pin':11, 'axis':'D', 'motor_spd':300000},
        }
    rpins = {key['rpi_pin'] : name for name, key in mconf.items()}

    # parse user args
    par = argparse.ArgumentParser(description=doc)
    arg, st, sf = par.add_argument, "store_true", "store_false"

    # motor functions
    arg('--steps', nargs='*', help="get steps to move [motor_name] a [value]")
    arg('--zero', nargs=1, type=str, help="zero motor: [source,linear,rotary]")
    arg("--move", nargs='*', help="move [motor_name] a distance [value]")

    # encoder functions & settings
    arg('-re', '--read_enc', nargs=1, type=int, help='read encoder at rpi pin')
    arg('-ze', '--zero_enc', nargs=1, type=int, help='zero encoder at rpi pin')

    # settings
    arg('--config', action=st, help='print hardware connection credentials')
    arg('--status', action=st, help='print current status of the motor system')
    arg('-a', '--angle_check', nargs=1, type=float, help="set encoder check angle")
    arg('-v', '--verbose', action=st, help='set verbose output')
    arg('-s', '--t_sleep', nargs=1, type=float, help='set sleep time')
    arg('-c', '--com_spd', nargs=1, type=int, help='set RPi comm speed (Hz)')
    arg('-m', '--max_reads', nargs=1, type=int, help='set num. tries to zero encoder')
    arg('--constraints', action=sf, help="DISABLE constraints (bad idea!)")

    args = vars(par.parse_args())

    # override default options
    t_sleep = args['t_sleep'][0] if args['t_sleep'] else 0.01 # sec
    com_spd = args['com_spd'][0] if args['com_spd'] else 100000  # Hz
    max_reads = args['max_reads'][0] if args['max_reads'] else 3 # tries
    angle_check = args['angle_check'][0] if args['angle_check'] else 180 # degrees
    verbose = args['verbose'] # overall verbosity (currently T/F)
    constraints = args['constraints'] # DISABLE motor step checks (DON'T!)

    # update motor config (variable names must match)
    for key, val in locals().items():
        if key in args:
            mconf[key] = val

    # ==========================================================================
    connect_to_controller(verbose) # check Newmark and RPi by default

    if args['config']:
        connect_to_controller(verbose=True)
        print("\nCredentials:")
        pprint(ipconf)
        print("\nCAGE motor system settings:")
        pprint(mconf)

    if args['status']:
        # TODO: add more here, like print(df_history)
        check_limit_switches(verbose=True)

    if args['read_enc']:
        rpi_pin = int(args['read_enc'][0])
        query_encoder(rpi_pin, t_sleep, com_spd, verbose)

    if args['zero_enc']:
        rpi_pin = int(args['zero_enc'][0])
        query_encoder(rpi_pin, t_sleep, com_spd, verbose, max_reads, zero=True)

    if args['steps']:
        # Check steps calculation w/o actually moving motors
        motor_name = args['steps'][0]
        input_val = float(args['steps'][1])
        get_steps(motor_name, input_val, angle_check, constraints, verbose)

    if args['move']:
        # Actually move motors
        motor_name = args['move'][0]
        input_val = float(args['move'][1])
        move_motor(motor_name, input_val, angle_check, constraints, verbose)

    if args['zero']:
        motor_name = args['move'][0]
        zero_motor(motor_name, angle_check, verbose)

    # if args['center']:
    #     motor_name = args['move'][0]
    #     center_motor(motor_name, angle_check, verbose)


def connect_to_controller(verbose=True):
    """
    connect to the Newmark motor controller and ping the RPi
    """
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

    if verbose:
        print("\nConnected to Newmark controller.")
        print(f"  {gp.GInfo()}")
        print(f"  {gp.GVersion()}")

    # now ping the CAGE RPi
    ping_rpi = os.system(f"ping -c 1 {ipconf['cage_rpi_ipa']} >/dev/null 2>&1")
    if ping_rpi == 0:
        if verbose:
            print(f"\nCAGE RPi at {ipconf['cage_rpi_ipa']} is online.")
    else:
        print("CAGE RPi isn't active on the network!  Beware ...")


def check_limit_switches(verbose=True):
    """
    Check the current status of the 4 limit switches.
    NEWMARK CONVENTION: True == 1 == OFF, False == 0 == ON
    """
    src = mconf['source']['axis']
    lfw = mconf['linear']['axis']
    lrv = mconf['linear']['axis']
    rot = mconf['rotary']['axis']

    # run gclib cmds and cast state to bools
    b_source = bool(float(gc(f"MG _LF {src}")))
    b_linear_fwd = bool(float(gc(f"MG _LF {lfw}")))
    b_linear_rev = bool(float(gc(f"MG _LR {lrv}")))
    b_rotary = bool(float(gc(f"MG _LF {rot}")))

    source = "OFF" if b_source else "ON"
    linear_fwd = "OFF" if b_linear_fwd else "ON"
    linear_rev = "OFF" if b_linear_rev else "ON"
    rotary = "OFF" if b_rotary else "ON"

    if verbose:
        print("\nLimit switch status:")
        print(f"  Source motor:      {source}")
        print(f"  Linear motor fwd:  {linear_fwd}")
        print(f"  Linear motor rev:  {linear_rev}")
        print(f"  Rotary motor:      {rotary}")

    # return the labels instead of the bools
    return source, linear_fwd, linear_rev, rotary


def query_encoder(rpi_pin, t_sleep=0.01, com_spd=10000, verbose=True, max_reads=3, zero=False):
    """
    Access the RPi-side routine, "read_encoders.py" via SSH.
    To read an encoder position:
        $ python3 motor_movement.py -p [rpi_pin] [options: -s, -c, -v]
    To zero an encoder:
        $ python3 motor_movement.py -z [rpi_pin] [options: -m, -s, -c, -v]
    """
    shell = spur.SshShell(hostname=ipconf["cage_rpi_ipa"],
                          username=ipconf["cage_rpi_usr"],
                          password=ipconf["cage_rpi_pwd"])
    with shell:
        cmd = "python3 cage/motors/read_encoders.py"
        if not zero:
            cmd += f" -p {rpi_pin} -s {t_sleep} -c {com_spd}"
        else:
            cmd += f" -z {rpi_pin} -m {max_reads} -s {t_sleep} -c {com_spd}"
        if verbose:
            cmd += " -v"

        # send the command and decode the output
        tmp = shell.run(shlex.split(cmd))
        ans = tmp.output.decode("utf-8")

    if verbose:
        enc_name = rpins[rpi_pin]
        if zero:
            print(f"\nZeroing {enc_name} encoder (pin {rpi_pin}).\nRPi output:")
        else:
            print(f"\nReading {enc_name} encoder position (pin {rpi_pin}).\nRPi output:")
        print(ans)

    return ans


def get_steps(motor_name, input_val, angle_check=180, constraints=True, verbose=False):
    """
    python motor_movement.py --step [name] [input_val] [options]

    50000 is the number of MOTOR steps for a full revolution.
    By default, we divide this by two and check the encoder position every 180
    degrees during a move.  Here, we take a desired move and calculate the number
    of 'cycles,' or number of times we want to stop the motion and check the
    encoder value.

    NOTES:
    - It's kind of arbitary that we check the encoder every 180 degrees.
      We could check it every 360 degrees.  Just not more than 360 because of
      the type of encoder we bought.
    - Direction convention: +1 is forward, -1 is backward.
    - Angles are relative in this code -- WE ASSUME YOU'VE JUST ZEROED THE MOTORS.

    TODO:
    - Put in "absolute" checks by using a history DataFrame.
    """
    step_check = 50000 * angle_check / 360

    if motor_name == "source":
        move_type = "degrees"
        direction = np.sign(input_val)
        n_steps = input_val * 50000 / 360 # 1:1 gear ratio
        if input_val < -25 and constraints:
            print(f"Angle {input_val} not allowed!  Source can't be rotated > 25 deg past normal incidence.")
            print("\nI can't believe you picked that.  How dare you!  jk ;-)")
            exit()

    elif motor_name == "linear":
        move_type = "mm"
        direction = np.sign(input_val)
        n_steps = input_val * 31573
        if input_val > 36 and constraints:
            print(f"Length {input_val} not allowed, you're gonna hit the detector!")
            exit()

    elif motor_name == "rotary":
        move_type = "degrees"
        direction = np.sign(input_val)
        n_steps = input_val * 50000 / 360 * 90 # 90:1 gear ratio
        if abs(input_val) > 330:
            print(f"Angle {input_val} is too big!  No angles greater than 330.")
            print("\nI can't believe you picked that.  How dare you!  jk ;-)")
            exit()

    # calculate how many times to check the encoders (default: every 180 degrees)
    n_cycles, r_steps = divmod(n_steps, step_check)
    n_cycles = abs(int(n_cycles))
    n_steps = int(n_steps)
    r_steps = int(direction * r_steps) # don't forget the remainder!

    if verbose:
        print(f"\nReady to move {motor_name} motor {input_val} {move_type}"
              f"\n  Will check encoder every {angle_check} degrees"
              f"\n  Steps: {n_steps}  n_cycles: {n_cycles}  remainder: {r_steps}")

    return {"dir": direction,
            "n_cycles": n_cycles,
            "n_steps": n_steps,
            "r_steps": r_steps,
            "step_check": step_check,
            "move_type": move_type,
            "input_val": input_val}


def move_motor(motor_name, input_val, angle_check=180, constraints=True, verbose=False):
    """
    $ python motor_movement.py --move [name] [input_val] [options]
    """
    # calculate the number of steps to move
    steps = get_steps(motor_name, input_val, angle_check, constraints, verbose)

    # zero the encoder (measure relative motion)
    result = query_encoder(mconf[motor_name]['rpi_pin'], mconf['t_sleep'],
                           mconf['com_spd'], verbose, mconf['max_reads'],
                           zero=True)
    zeroed = bool(result.split("\n")[-2].split(" ")[2]) # ugly string parse
    if not zeroed:
        print("ERROR! read_encoders was unable to zero the encoder.")
        exit()
    if verbose:
        print(f"Zeroed {motor_name} encoder.")

    # send initial setup commands to the controller
    axis = mconf[motor_name]['axis']
    mspd = mconf[motor_name]['motor_spd']
    gc('AB')
    gc('MO')
    gc(f'SH{axis}')
    gc(f'SP{axis}={mspd}')
    gc(f'DP{axis}=0')
    gc(f'AC{axis}={mspd}')
    gc(f'BC{axis}={mspd}')

    # useful counters
    n_checks = int(360 / angle_check)
    i_check = 1
    enc_tol = 10
    enc_fail = False
    n_cyc = abs(steps['n_cycles'])
    desired = steps['n_steps']
    moved = 0

    # start movement
    print("Beginning move ...")
    try:
        for i_cyc in range(1, n_cyc+1):

            # goal: move this many motor steps
            move = int(steps['dir'] * steps['step_check'])

            # take current reading of encoder position (quiet)
            enc_pos = int(query_encoder(mconf[motor_name]['rpi_pin'],
                                        mconf['t_sleep'], mconf['com_spd'],
                                        verbose=False).rstrip())

            # NOTE: add actual motion
            pct = 100 * abs(moved/desired)
            print(f"{i_cyc}/{n_cyc}  attempting: {move}/{moved}  ({pct:.1f}%)  encoder: {enc_pos}  actual pos: XX steps['move_type']")

            # begin motion
            gc(f'PR{axis}={move}')
            gc(f'BG{axis}')
            gp.GMotionComplete(axis)

            # cross-check encoder position with motor step commands
            if constraints:
                enc_fail = False
                mod = i_cyc % n_checks
                exp_pos = int(mod * 2**14 / n_checks) # expected enc position
                # print(i_cyc, mod, exp_pos) # useful debug, don't delete this

                if mod == 0:
                    # full rotation (360 deg)
                    if (enc_pos > enc_tol) or (enc_pos < 2**14 - enc_tol):
                        enc_fail = True
                else:
                    # partial rotation
                    if not exp_pos - enc_tol < enc_pos < exp_pos + enc_tol:
                        enc_fail = True
                if enc_fail:
                    print("Encoder position check failed!\nAborting move ...")
                    break

            # increment total steps counter
            moved += move

        # move the final remainder
        if not enc_fail:
            move = steps['r_steps']
            gc(f'PR{axis}={move}')
            gc(f'BG{axis}')
            gp.GMotionComplete(axis)
            moved += move
            i_cyc += 1
            print(f'{i_cyc}/{n_cyc}  {move}  {moved}  {desired}  {pct:.2f}%')

    except gclib.GclibError:
        print("The motor probably hit the limit switch.")

    # show a final summary
    cap = motor_name.capitalize()
    input_val = steps['input_val']
    move_type = steps['move_type']

    # convert back
    if motor_name == "source":
        final_val = moved / (50000 / 360)
    if motor_name == "linear":
        final_val = moved / 31573
    if motor_name == "rotary":
        final_val = moved / (50000 / 360 * 90)

    # print final warning
    if not constraints:
        print("\nConstraints are OFF, the following summary is probably wrong:")

    # final summary.  TODO: save these results to DataFrame with iloc (ask Clint)
    print(f"\n{cap} motor movement summary:"
          f"\n  Attempted: {input_val} {move_type}"
          f"\n  Equivalent motor steps: {desired}"
          f"\n  Total steps moved: {moved}"
          f"\n  Final position: {final_val} {move_type}")


def zero_motor(motor_name, angle_check, verbose, constraints=True):
    """
    run the motors backwards (or forwards ;-) to their limit switches
    """
    zeros = {
        'source' : 360, # go FORWARDS 360 degrees (the full amt)
        'linear' : -51,  # the full backwards travel (2 inches)
        'rotary' : 360  # go FORWARDS 360 degrees (b/c of our convention)
        }
    move_motor(motor_name, zeros[motor_name])


def center_motor(motor_name, verbose, constraints=True):
    """
    here's where the user has to say "Y", etc.
    this function should be fairly simple and just call move_motor appropriately
    """
    # first, zero the motor
    # zero_motor(motor_name, angle_check=mconf['angle_check'], verbose)

    if motor_name == "linear":
        print("move the thing forward 3.175 mm")
        # move_motor("linear", etc.)
    if motor_name == "source":
        print('do the special limit checks')
        move_motor()
        # zero_motor("source",...)
    else:
        print("Other motors aren't special, leave me alone")


if __name__=="__main__":
    main()
