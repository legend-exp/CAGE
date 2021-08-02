#!/usr/bin/env python3
import sys, os
import time
import json
import shlex
import argparse
from argparse import RawTextHelpFormatter as rthf
import multiprocessing
import gclib
import spur
import numpy as np
from pprint import pprint
import pandas as pd
from datetime import datetime
sys.path.append('../sims')
from source_placement import *

def main():
    doc="""
    CAGE motor movement suite.

    REFERENCES:
    https://elog.legend-exp.org/UWScanner/200130_140616/cage_electronics.pdf
    http://www.galilmc.com/sw/pub/all/doc/gclib/html/python.html
    """
    # load configuration (uses globals, it's bad practice but who cares)
    global gp, gc, conf, mconf, rpins, history_df, f_history
    gp = gclib.py()
    gc = gp.GCommand
    with open('../config.json') as f:
        ipconf = json.load(f)
    with open('motor_config.json') as g:
        motorDB = json.load(g)
    mconf = motorDB["mconf"]
    rpins = {key['rpi_pin'] : name for name, key in mconf.items()}

    # TODO: change this when we want to do a new campaign (see motor_config.json)
    campaign_number = "8"

    # parse user args
    par = argparse.ArgumentParser(description=doc, formatter_class=rthf)
    arg, st, sf = par.add_argument, "store_true", "store_false"

    # motor functions
    arg('--steps', nargs='*', help="get steps to move [motor_name] a [value]")
    arg('--zero', nargs=1, type=str, help="zero motor: [source,linear,rotary]")
    arg("--move", nargs='*', help="move [motor_name] a distance [value]")
    arg("--beam_pos_move", nargs=1, help="Use source_placement.py to calculate motor movements for [oppi] or [icpc]")
    arg("--center", nargs='*', help="center source or linear motor")

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
    arg('--history', action=sf, help="zero using full range of motor motion rather than using history")
    arg('--init_df', action=st, help='initialize the motor history dataframe')
    arg('--show_df', action=st, help='print motor history df and exit')

    args = vars(par.parse_args())

    # override default options
    t_sleep = args['t_sleep'][0] if args['t_sleep'] else 0.01 # sec
    com_spd = args['com_spd'][0] if args['com_spd'] else 100000  # Hz
    max_reads = args['max_reads'][0] if args['max_reads'] else 3 # tries
    angle_check = args['angle_check'][0] if args['angle_check'] else 180 # degrees
    verbose = args['verbose'] # overall verbosity (currently T/F)
    constraints = args['constraints'] # DISABLE motor step checks (DON'T!)
    history = args['history'] # zero motors using full range of motion

    # update motor config (variable names must match)
    for key, val in locals().items():
        if key in args:
            mconf[key] = val

    # load the motor movement history dataframe for this campaign
    f_history = f"./history/motorhistory_{motorDB['campaign'][campaign_number][0]}.h5"
    try:
        history_df = pd.read_hdf(f_history)
    except:
        print(f'ERROR: Cannot find {f_history}')
        ans = input(f'Would you like to initialize {f_history}? y/n \n -->')
        if ans == 'y':
            init_history_df()
        exit()

    cp_label = motorDB['campaign'][campaign_number][0]
    print(f"Current campaign: {campaign_number} : cp_label")
    print(f'Move history is saved in: {f_history}')

    if args['show_df']:
        show_history(f_history)
        exit()

    # =====================================================================
    connect_to_controller(verbose, ipconf) # check Newmark and RPi by default

    if not constraints:
        ans = input('WARNING: Are you sure you want constraints off? Danger of damaging system! y/n \n -->')
        ans = ans.lower()
        if ans != 'y':
            exit()

    if args['config']:
        connect_to_controller(verbose=True, ipconf=ipconf)
        print("\nCredentials:")
        pprint(ipconf)
        print("\nCAGE motor system settings:")
        pprint(mconf)

    if args['init_df']:
        init_history_df()

    if args['status']:
        check_limit_switches(verbose=True)
        print(history_df)
        # print(history_df.to_string())

    if args['read_enc']:
        rpi_pin = int(args['read_enc'][0])
        query_encoder(rpi_pin, t_sleep, com_spd, verbose, ipconf=ipconf)

    if args['zero_enc']:
        rpi_pin = int(args['zero_enc'][0])
        query_encoder(rpi_pin, t_sleep, com_spd, verbose, max_reads, zero=True,
                      ipconf=ipconf)

    if args['steps']:
        # Check steps calculation w/o actually moving motors
        motor_name = args['steps'][0]
        input_val = float(args['steps'][1])
        get_steps(motor_name, input_val, angle_check, constraints, verbose)

    # --- Make sure lift interlock is engaged ---
    lift_interlock(ipconf)

    # Actually move motors!
    if args['move']:
        motor_name = args['move'][0]
        input_val = float(args['move'][1])
        approve_move(history_df, motor_name, input_val, False, constraints)
        move_motor(motor_name, input_val, history_df, angle_check, constraints, verbose)

    if args['zero']:
        motor_name = args['zero'][0]
        zero_motor(motor_name, angle_check, history_df, verbose, constraints, history)

    if args['center']:
        motor_name = args['center'][0]
        approve_move(history_df, motor_name, 0, True, constraints)
        center_motor(motor_name, angle_check, history_df, verbose, constraints)

    # incorporating Joule's source placement code--TEST
    if args['beam_pos_move']:
        source_amount, linear_amount = beam_pos_move(args['beam_pos_move'][0])

        approve_move(history_df, 'linear', linear_amount, False, constraints)
        move_motor('linear', linear_amount, history_df, angle_check, constraints, verbose)

        approve_move(history_df, 'source', source_amount, False, constraints)
        move_motor('source', -1*source_amount, history_df, angle_check, constraints, verbose)
        exit()


def show_history(f_history):
    """
    $ python3 motor_movement.py --show_df
    Show the recent move history.
    """
    df = pd.read_hdf(f_history)
    print(df.columns)
    print(df)


def connect_to_controller(verbose=True, ipconf=None):
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


def lift_interlock(ipconf):
    """
    Access the RPi-side routine, "read_encoders.py" via SSH.
    To read an encoder position:
        $ python3 motor_movement.py -p [rpi_pin] [options: -s, -c, -v]
    To zero an encoder:
        $ python3 motor_movement.py -z [rpi_pin] [options: -m, -s, -c, -v]
    """
    shell = spur.SshShell(hostname=ipconf["pressure_rpi_ipa"],
                          username=ipconf["pressure_rpi_usr"],
                          password=ipconf["pressure_rpi_pwd"])
    with shell:
        result = shell.run(["python3", "/home/pi/cage/motors/lift_interlock.py"])

    result = int(result.output.decode("utf-8"))

    if result != 1:
        print("WARNING: Rack and Pinion is not lifted to safe distance")
        print("Lift rack and pinion and place motor movement block so that pressure pad is engaged")
        print("Then retry your command")
        exit()
    else:
        print(f'Lift interlock : {result}, motor movements are allowed.')


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


def query_encoder(rpi_pin, t_sleep=0.01, com_spd=10000, verbose=True, max_reads=3, zero=False, ipconf=None):
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

    # print(angle_check/360)

    if motor_name == "source":
        move_type = "degrees"
        direction = np.sign(input_val)
        n_steps = input_val * 50000 / 360 # 1:1 gear ratio

    elif motor_name == "linear":
        move_type = "mm"
        direction = np.sign(input_val)
        n_steps = input_val * 31496

    elif motor_name == "rotary":
        move_type = "degrees"
        direction = np.sign(input_val)
        n_steps = input_val * 50000 / 360 * 90 # 90:1 gear ratio
        # if abs(input_val) > 330 and constraints:
        #     print(f"Angle {input_val} is too big!  No angles greater than 330.")
        #     print("\nI can't believe you picked that.  How dare you!  jk ;-)")
        #     exit()

    # calculate how many times to check the encoders (default: every 180 degrees)
    n_cycles, r_steps = divmod(direction * n_steps, step_check)
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


def move_motor(motor_name, input_val, history_df, angle_check=180, constraints=True, verbose=False, zero=False):
    """
    $ python motor_movement.py --move [name] [input_val] [options]

    TODO: pass failure mode (encoder check fail, motor controller off, etc)
    to the history dataframe as a column.
    Running total of the slip
    """
    # calculate the number of steps to move
    steps = get_steps(motor_name, input_val, angle_check, constraints, verbose)

    # init this row in the history DataFrame (move_completed = False)
    update_history(motor_name, 0, 0, zero, steps, move_completed=False)

    # zero the encoder (measure relative motion)
    with open('../config.json') as f:
        ipconf = json.load(f)
    result = query_encoder(mconf[motor_name]['rpi_pin'], mconf['t_sleep'],
                           mconf['com_spd'], verbose, mconf['max_reads'],
                           zero=True, ipconf=ipconf)
    zeroed = bool(result.split("\n")[-2].split(" ")[2]) # ugly string parse
    if not zeroed:
        print("ERROR! read_encoders was unable to zero the encoder.")
        exit()
    if verbose:
        print(f"Zeroed {motor_name} encoder.")

    pool = multiprocessing.Pool(1)
    try:
        result = pool.apply_async(run_motion, args=(motor_name, mconf, ipconf, angle_check, steps, constraints), error_callback=ecb)
        pool.close()
        pool.join()

    except KeyboardInterrupt:
        # emergency stop!
        pool2 = multiprocessing.Pool(1)
        pool2.apply_async(emergency_stop, args=(motor_name, mconf, ipconf),
                          error_callback=ecb)
        pool.terminate()
        pool2.close()
        pool2.join()
        result = None

    if result is not None:
        n_desired, n_moved, enc_fail = result.get(timeout=1)
    else:
        n_desired = steps['n_steps']
        n_moved, enc_fail = 0, True

    # convert back to physical units
    if motor_name == "source":
        relative_pos = n_moved / (50000 / 360)
    if motor_name == "linear":
        relative_pos = n_moved / 31496
    if motor_name == "rotary":
        relative_pos = n_moved / (50000 / 360 * 90)

    # print final warning
    if not constraints:
        print("\nConstraints are OFF, the following summary is probably wrong:")

    # print a final summary
    cap = motor_name.capitalize()
    input_val = steps['input_val']
    move_type = steps['move_type']
    print(f"\n{cap} motor movement summary:"
          f"\n  Attempted: {input_val} {move_type}"
          f"\n  Equivalent motor steps: {n_desired}"
          f"\n  Total steps moved: {n_moved}"
          f"\n  Final position: {relative_pos} {move_type}")

    move_complete = True
    if constraints and enc_fail==True:
        move_complete = False

    # update the DF with a successful move.
    update_history(motor_name, relative_pos, n_moved, zero, steps, move_completed=move_complete)


def ecb(this):
    """ helper function for move_motor. """
    print('error callback:', this)


def run_motion(motor_name, mconf, ipconf, angle_check, steps, constraints):
    """
    called by move_motor.  must be global for multiprocessing to work.
    """
    # setup and counters
    axis = mconf[motor_name]['axis']
    mspd = mconf[motor_name]['motor_spd']
    enc_tol = mconf[motor_name]["enc_slip"]
    mip = ipconf['newmark']
    n_checks = int(360 / angle_check)
    i_check = 1
    enc_fail = False
    n_cyc = abs(steps['n_cycles'])
    n_desired = steps['n_steps']
    n_moved = 0
    enc_pos = 0

    # connect to controller
    gp = gclib.py()
    gc = gp.GCommand
    try:
        gp.GOpen(f"{mip} --direct")
    except:
        print("ERROR: couldn't connect to Newmark controller!")

    # send initial setup commands to the controller
    gc('AB')
    gc('MO')
    gc(f'SH{axis}')
    gc(f'SP{axis}={mspd}')
    gc(f'DP{axis}=0')
    gc(f'AC{axis}={mspd}')
    gc(f'BC{axis}={mspd}')

    print("Beginning move ...")
    try:
        for i_cyc in range(1, n_cyc+1):

            # goal: move this many motor steps
            n_move = int(steps['dir'] * steps['step_check'])

            # NOTE: add actual motion
            pct = 100 * abs(n_moved/n_desired)

            print(f"{i_cyc}/{n_cyc}  attempting: {n_move}/{n_moved}  {pct:.1f}%  encoder: {enc_pos:6}  actual pos: XX {steps['move_type']}")

            # begin motion
            gc(f'PR{axis}={n_move}')
            gc(f'BG{axis}')
            gp.GMotionComplete(axis)
            time.sleep(.1)

            # take current reading of encoder position (quiet)
            enc_pos = int(query_encoder(mconf[motor_name]['rpi_pin'],
                                        mconf['t_sleep'], mconf['com_spd'],
                                        verbose=False, ipconf=ipconf).rstrip())
            enc_pos2 = int(query_encoder(mconf[motor_name]['rpi_pin'],
                                        mconf['t_sleep'], mconf['com_spd'],
                                        verbose=False, ipconf=ipconf).rstrip())

            # cross-check encoder position with motor step commands
            if constraints:
                enc_fail = False
                mod = i_cyc % n_checks
                exp_pos = int(mod * 2**14 / n_checks) # expected enc position

                if mod == 0:
                    # print("full rotation --- ", exp_pos, enc_tol, enc_pos)
                    if (enc_pos > enc_tol) and (enc_pos < 2**14 - enc_tol):
                        enc_fail = True
                else:
                    # print("partial rotation --- ", exp_pos, enc_tol, enc_pos)
                    if not (exp_pos - enc_tol) < enc_pos < (exp_pos + enc_tol) and not (exp_pos - enc_tol) < enc_pos2 < (exp_pos + enc_tol):
                        enc_fail = True
                if enc_fail:
                    print("Encoder position check failed!\nAborting move ...")
                    type = "full" if mod==0 else "partial"
                    print(f"{type} rotation --\n Expected pos: {exp_pos}"
                          f"\n Encoder tolerance: {enc_tol}"
                          f"\n Encoder position: {enc_pos}")
                    print("NOTE: if you just zeroed the motor, this is probably OK.  Check the limit switch.")
                    break

            # increment total steps counter
            n_moved += n_move

        # move the final remainder
        if not enc_fail:
            r_steps = steps['r_steps']

            gc(f"PR{axis}={r_steps}")
            gc(f'BG{axis}')
            gp.GMotionComplete(axis)

            n_moved += steps['r_steps']

            enc_pos = int(query_encoder(mconf[motor_name]['rpi_pin'],
                                        mconf['t_sleep'], mconf['com_spd'],
                                        verbose=False, ipconf=ipconf).rstrip())

            print(f"final move: {r_steps}/{n_moved}  encoder: {enc_pos:6}  actual pos: XX {steps['move_type']}")

    except gclib.GclibError:
        print("gclib error: The limit switch is engaged.")
        # NOTE: this creates two "False" rows in the DataFrame.  which is OK.
        gp.GMotionComplete(axis)

    return n_desired, n_moved, enc_fail


def emergency_stop(motor_name, mconf, ipconf):
    """
    called by move_motor.  must be global for multiprocessing to work,
    and requires we set up a whole separate communication session (because
    this is called in an independent Python thread.)
    """
    print('\nGot Emergency Stop signal!  Attempting to stop motion ...')

    mip = ipconf['newmark']
    axis = mconf[motor_name]['axis']

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

    print("WARNING: You've just used Emergency Stop.  The motor may still be\n",
          "   'humming', i.e. powered but not moving.  Sometimes this is\n",
          "   audible, but not always. You can either try \n",
          "   another (safe) motion to reset it, or reset the Newmark\n",
          "   motor controller.")


def beam_pos_move(detector):


    radial_pos = float(input("Desired radial position of beam \n -->"))
    source_angle = float(input("Desired source angle with respect to detector surface \n -->"))

    if detector == 'icpc':
        source_rot, linear_move = positionCalc(radial_pos, source_angle)
    elif detector == 'oppi':
        source_rot, linear_move = positionCalc(radial_pos, source_angle, icpc=False)
    else:
        print("That is not a valid detector, please choose icpc or oppi")
        exit()

    return source_rot, linear_move


def zero_motor(motor_name, angle_check, history_df, verbose, constraints=True, history=True):
    """
    run the motors backwards (or forwards) to their limit switches
    """
    zeros = {
        'source' : 360, # go FORWARDS 360 degrees (the full amt)
        'linear' : -51,  # the full backwards travel (2 inches)
        'rotary' : 360  # go FORWARDS 360 degrees (b/c of our convention)
        }
    if history:
        zeros = {
                'source' : -1*history_df.iloc[-1,:][f'source_total'] + 1,
                'linear' : -1*history_df.iloc[-1,:][f'linear_total'] - 1,
                'rotary' : -1*history_df.iloc[-1,:][f'rotary_total'] + 1
        }
    move_motor(motor_name, zeros[motor_name], history_df, angle_check,
                constraints, verbose, zero=True)


def center_motor(motor_name, angle_check, history_df, verbose, constraints=True):
    """
    here's where the user has to say "Y", etc.
    this function should be fairly simple and just call move_motor appropriately
    """
    # first, zero the motor
    # zero_motor(motor_name, angle_check=mconf['angle_check'], verbose)

    if motor_name == "linear":
        print("move the thing forward 3.175 mm")
        move_motor("linear", 3.175, history_df, angle_check, constraints, verbose)
    elif motor_name == "source":
        print('do the special limit checks')
        move_motor("source", -180, history_df, angle_check, constraints, verbose)
        # zero_motor("source",...)
    else:
        print("Other motors aren't special, leave me alone")


def init_history_df():
    """
    Initialize the motor movement history dataframe in h5 format
    """
    print('WARNING: You are about to create a new motor history dataframe.')
    print('Current file can be found at:', f_history)
    ans = input('Are you sure you want to do this? y/n \n -->')
    if ans != 'y':
        print('Cool')
        exit()

    init_columns = [
        'motor_name', 'move_completed', 'distance_steps', 'distance_real', 'move_type',
        'source_total', 'linear_total', 'rotary_total', 'timestamp']
    ts = pd.Timestamp(datetime.utcnow())
    init_values = [['Bob_motor', False, 0, 0, 'angle', 0, 0, 0, ts]]


    df = pd.DataFrame(init_values, columns=init_columns)
    print(df)
    df.to_hdf(f"{f_history}", key="motor_history", mode='w', format='table')


def approve_move(history_df, motor_name, input_val, center=False, constraints=True):
    """
    NOTICE: This is where the majority of ANGLE / LENGTH constraints go!

    Make sure no motors hit anything they shouldn't.  Use constraints relative to
    zero positions, then query history dataframe to subtract off current positions
    from constraints.

    NOTE:
    If we need to remember how these constraints were picked, see elog:
    https://elog.legend-exp.org/UWScanner/145
    """
    source_constraint = mconf["source"]["constraint"]
    linear_constraint = mconf["linear"]["constraint"]
    rotary_constraint = mconf["rotary"]["constraint"]

    if not constraints:
        print('WARNING: MOTORS CAN HIT LIMITS AND DAMAGE PARTS AND MAKE US SAD')

    if constraints:

        # Handle centering for linear and source motors
        if center:
            check_zero = history_df.iloc[-1,:][f'{motor_name}_total']
            if check_zero != 0:
                print(f'ERROR: Cannot center {motor_name} motor if last move was not zeroing the motor, because you could go out of range.  See elog 145')
                print('Please zero motor first')
                exit()

        # Handle all other motions (zeroing and arbitrary moves)
        if not center:
            if motor_name == 'source':

                # TODO: switch to DataFrame, and only ask the user a question if the DF fails
                # ans = input('Did you just park (ang=180) the source motor? y/n \n -->')
                # ans = ans.lower()
                # if ans != 'n':
                #     print('Please center (ang=0) the source motor before a movement')
                #     exit()

                source_constraint = -(history_df.iloc[-1,:]['source_total'] - source_constraint)
                if input_val < source_constraint:
                    print('WARNING: CANNOT DO THIS MOVE\n Source motor switch will hit encoder')
                    print(f'Only {source_constraint} degree movement away from center position allowed')
                    exit()

            if motor_name == 'linear':

                # TODO: switch to DataFrame, and only ask the user a question if the DF fails
                # ans = input('Did you just move the linear motor to its 0 position at 3.175 mm from switch? y/n \n -->')
                # ans = ans.lower()
                # if ans != 'y':
                #     print('Please set linear motor to its 0 position')
                #     exit()

                linear_constraint = linear_constraint - history_df.iloc[-1,:]['linear_total']
                if input_val > linear_constraint:
                    print(f'WARNING: CANNOT DO THIS MOVE\n Linear motor cannot exceed {linear_constraint} mm movement')
                    exit()

            if motor_name == 'rotary':
                rotary_constraint = -(history_df.iloc[-1,:]['rotary_total'] - rotary_constraint)
                if input_val < rotary_constraint:
                    print(f'WARNING CANNOT DO THIS MOVE \n Rotary motion cannot be more negative than {rotary_constraint} degrees')
                    exit()


def update_history(motor_name, relative_pos, n_moved, zero, steps, move_completed=False):

    df_idx, df_ncols = history_df.shape

    last_row = history_df.iloc[-1].copy()

    if zero:
        running_total = 0
    if not zero:
        running_total = relative_pos + last_row.loc[f'{motor_name}_total']

    new_vals = {
        "motor_name":motor_name,
        "distance_real": relative_pos,
        "distance_steps": n_moved,
        "move_completed": move_completed,
        "move_type": steps["move_type"],
        f'{motor_name}_total': running_total,
        "timestamp":pd.Timestamp(datetime.utcnow())
    }
    for k, v in new_vals.items():
        last_row[k] = v

    # create new row when move starts, and update values when complete
    if move_completed:
        history_df.loc[df_idx] = last_row # update the last row
    else:
        history_df.loc[df_idx+1] = last_row # append a new row

    # save_to_file
    history_df.to_hdf(f_history, key="motor_history")


if __name__=="__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("hahaha")
