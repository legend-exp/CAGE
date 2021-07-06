#!/usr/bin/env python3
import sys
import argparse
import subprocess as sp

def main():
    doc="""
    control job submission to the NERSC Cori batch system.
    Example usage (note enclosing double quote):
    
    $ ./run_slurm.py --cmd "python3 example.py --q 'cycle>438'"
    
    TODO: set up job pump mode?
    """
    rthf = argparse.RawTextHelpFormatter
    par = argparse.ArgumentParser(description=doc, formatter_class=rthf)
    arg, st, sf = par.add_argument, 'store_true', 'store_false'
    arg('--cmd', nargs=1, type=str, help="cmd to execute on cluster")
    args = par.parse_args()
    
    cmd = args.cmd[0]
    submit_batch_job(cmd)


def submit_batch_job(cmd):
    """
    
    """
    sbatch_opts = [
        f"--chdir=/global/project/projectdirs/legend/software/CAGE/processing",
        f"--output=/global/project/projectdirs/legend/software/CAGE/processing/logs/cori-%j.txt",
        "--image=legendexp/legend-base:latest",
        "-C haswell",
        "--account m2676",
        # "--export=A=$A,b=$b"
        "--export=HDF5_USE_FILE_LOCKING=FALSE",
        "--qos=debug",
        # "--qos=regular",
        # "--qos=shared",
        "-t 00:30:00"
        # "-t 36:00:00"
        # "-t 24:00:00"
    ]
    sbatch_str = " ".join([str(s) for s in sbatch_opts])
    batch_cmd = f"sbatch {sbatch_str} slurm.slr \"{cmd}\""
    # print(batch_cmd)
    sp.call(batch_cmd, shell=True)    


if __name__=="__main__":
    main()