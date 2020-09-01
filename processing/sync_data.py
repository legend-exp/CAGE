#!/usr/bin/env python3
import os
import glob
import json
from datetime import datetime
import subprocess as sp
from pprint import pprint
from pygama.utils import *

def main():
    """
    sync CAGE data with NERSC.
    """
    with open("cage.json") as f:
        expDB = json.load(f)

    # run_rsync(expDB)
    daq_cleanup(expDB)


def run_rsync(expDB, test=False):
    """
    NOTE: this doesn't work yet for wisecg, file permissions
    at the source directory appear to be messed up.  Need to
    talk with Robert Varner to fix.
    """
    if "mjcenpa" not in os.environ["USER"]:
        print("Error, we're not on the MJ60 DAQ machine.  Exiting ...")
        exit()

    daq_dir = os.path.expandvars(expDB["daq_dir"] + "/")
    daq_nersc = "{}:{}/".format(expDB["nersc_login"], expDB["nersc_dir"])

    if test:
        cmd = "rsync -avh --dry-run {} {}".format(daq_dir, daq_nersc)
    else:
        # cmd = "rsync -avh --no-perms {} {}".format(daq_dir, daq_nersc)
        # try to fix permissions issue?
        cmd = "rsync -avh --no-o --no-g --no-perms {} {}".format(daq_dir, daq_nersc)
    sh(cmd)


def daq_cleanup(expDB):
    """
    build a list of files on the DAQ and nersc, check integrity,
    and delete files on the DAQ only if we're sure the transfer was successful.
    """
    if "mjcenpa" not in os.environ["USER"]:
        print("Error, we're not on the MJ60 DAQ machine.  Exiting ...")
        exit()

    # local (DAQ) list
    datadir_loc = os.path.expandvars(expDB["daq_dir"] + "/")
    filelist_loc = glob.glob(datadir_loc + "/**", recursive=True)
    # for f in filelist_loc:
        # print(f)

    # remote list
    # args = ['ssh', expDB['nersc_login'], 'ls -R '+expDB["nersc_dir"]]
    args = ['ssh', expDB['nersc_login'], 'ls -R '+expDB['nersc_dir']]
    ls = sp.Popen(args, stdout=sp.PIPE, stderr=sp.PIPE)
    out, err = ls.communicate()
    out = out.decode('utf-8')
    filelist_nersc = out.split("\n")
    filelist_nersc = [f for f in filelist_nersc if ":" not in f and len(f)!=0]
    # for f in filelist_nersc:
        # print(f)

    # make sure all files have successfully transferred
    for f in filelist_loc:
        fname = f.split("/")[-1]
        if len(fname) == 0:
            continue
        if fname not in filelist_nersc:
            print("whoa, ", fname, "not found in remote list!")
            exit()

    print("All files in:\n    {}\nhave been backed up to NERSC."
          .format(datadir_loc))
    print("It should be OK to delete local files.")

    # don't delete these files, orca needs them
    ignore_list = [".Orca", "RunNumber"]

    # now delete old files, ask for Y/N confirmation
    ans = input("OK to delete local files? y/n:")
    if ans.lower() == 'y':
        for f in filelist_loc:
            f.replace(" ", "\ ")
            if os.path.isfile(f):
                if any(ig in f for ig in ignore_list):
                    continue
                print("Deleting:", f)
                os.remove(f)

    now = datetime.now()
    print("Processing is up to date!", now.strftime("%Y-%m-%d %H:%M"))



if __name__=="__main__":
    main()
