import h5py
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from pygama.analysis.peak_fitting import gauss_mode_width_max
from pygama import DataGroup, lh5
import jump_detection as jd
import gain_relaxation as rd


#Parameters: query to fileDB
#Returns: table or list or dictionary of cycle number and calibratable (true/false)
def main():
    doc="""
    Find cycles where the gain changes to exclude from energy calibration
    """

    rthf = argparse.RawTextHelpFormatter
    par = argparse.ArgumentParser(description=doc, formatter_class=rthf)
    arg, st, sf = par.add_argument, 'store_true', 'store_false'
    arg('-q', '--query', nargs=1, type=str,
        help="select file group to calibrate: -q 'run==1' ")
    arg('-u', '--user', action='store_true',
        help="use lh5 user directory")
    args = par.parse_args()
    user = args.user

    # load main DataGroup, select files from cmd line
    dg = DataGroup('$CAGE_SW/processing/cage.json', load=True)
    if args.query:
        que = args.query[0]
        dg.fileDB.query(que + "and skip==False", inplace=True)
    else:
        dg.fileDB = dg.fileDB[-1:]
    
    global lh5_dir
    if user:
        lh5_dir = os.path.expandvars(dg.lh5_user_dir)
    else:
        lh5_dir = os.path.expandvars(dg.lh5_dir)
        
    global time_intervals
    time_intervals = 300
    
    print(find_uncalibratable(dg))
    
    return 0
    
    
def find_uncalibratable(dg):
    jumps = jd.find_jumps(dg, False)
    drifts = rd.find_drifts(dg, False)
    
    uncalibratable = []
    
    for jump in jumps:
        if jump[1] not in uncalibratable:
            uncalibratable.append(jump[1])

    for drift in drifts:
        run = drift[0]
        cycle = dg[dg[f'run == {run}']]['cycle'].iloc[0] #Skip the first cycle of a run with drifts
        if cycle not in uncalibratable:
            uncalibratable.append(cycle)
            
    return uncalibratable

if __name__=="__main__":
    main()