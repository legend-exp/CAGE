#!/usr/bin/env python3
import sys, os, io
import json
import argparse
import numpy as np
import pandas as pd
from pprint import pprint
from pygama import DataSet

def main(argv):
    """
    Uses pygama's amazing DataSet class to process runs
    for different data sets and arbitrary configuration options
    defined in a JSON file.
    """
    run_db = './runDB.json'
    # -- parse args --
    par = argparse.ArgumentParser(description="data processing suite for MJ60")
    arg, st, sf = par.add_argument, "store_true", "store_false"
    arg("-ds", nargs='*', action="store", help="load runs for a DS")
    arg("-r", "--run", nargs=1, help="load a single run")
    arg("-t0", "--tier0", action=st, help="run daq_to_raw on list")
    arg("-t1", "--tier1", action=st, help="run raw_to_dsp on list")
    arg("-t", "--test", action=st, help="test mode, don't run")
    arg("-n", "--nevt", nargs='?', default=np.inf, help="limit max num events")
    arg("-i", "--ioff", nargs='?', default=0, help="start at index [i]")
    arg("-v", "--verbose", action=st, help="set verbose output")
    arg("-o", "--ovr", action=st, help="overwrite existing files")
    arg("-m", "--nomp", action=sf, help="don't use multiprocessing")
    args = vars(par.parse_args())

    # -- declare the DataSet --
    if args["ds"]:
        ds_lo = int(args["ds"][0])
        try:
            ds_hi = int(args["ds"][1])
        except:
            ds_hi = None
        ds = DataSet(ds_lo, ds_hi, md=run_db, v=args["verbose"])

    if args["run"]:
        ds = DataSet(run=int(args["run"][0]), md=run_db, v=args["verbose"])

    # -- start processing --
    if args["tier0"]:
        tier0(ds, args["ovr"], args["nevt"], args["verbose"], args["test"])

    if args["tier1"]:
        tier1(ds, args["ovr"], args["nevt"], args["ioff"], args["nomp"], args["verbose"],
              args["test"])


def tier0(ds, overwrite=False, nevt=np.inf, v=False, test=False):
    """
    Run daq_to_raw on a set of runs.
    [raw file] ---> [t1_run{}.lh5] (tier 1 file: basic info & waveforms)
    """
    from pygama.io.daq_to_raw import daq_to_raw

    for run in ds.runs:

        t0_file = ds.paths[run]["t0_path"]
        t1_file = ds.paths[run]["t1_path"]
        if t1_file is not None and overwrite is False:
            print("file exists, overwrite flag isn't set.  continuing ...")
            continue

        conf = ds.paths[run]["build_opt"]
        opts = ds.config["build_options"][conf]["tier0_options"]

        if test:
            print("test mode (dry run), processing Tier 0 file:", t0_file)
            print("writing to:", t1_file)
            continue

        daq_to_raw(t0_file, run, verbose=v, output_dir=ds.tier1_dir,
                     overwrite=overwrite, n_max=nevt, config=ds.config)#settings=opts)


def tier1(ds,
          overwrite=False,
          nevt=None,
          ioff=None,
          multiproc=True,
          verbose=False,
          test=False):
    """
    Run raw_to_dsp on a set of runs.
    [t1_run{}.h5] ---> [t2_run{}.h5]  (tier 2 file: DSP results, no waveforms)

    Can declare the processor list via:
    - json configuration file (recommended)
    - Intercom(default_list=True)
    - manually add with Intercom::add
    """
    from pygama.dsp.dsp_base import Intercom
    from pygama.io.raw_to_dsp import RunDSP

    for run in ds.runs:

        t1_file = ds.paths[run]["t1_path"]
        t2_file = ds.paths[run]["t2_path"]
        if t2_file is not None and overwrite is False:
            continue

        if test:
            print("test mode (dry run), processing Tier 1 file:", t1_file)
            continue

        conf = ds.paths[run]["build_opt"]
        proc_list = ds.config["build_options"][conf]["tier1_options"]
        proc = Intercom(proc_list)

        RunDSP(
            t1_file,
            proc,
            output_dir=ds.tier2_dir,
            overwrite=overwrite,
            verbose=verbose,
            multiprocess=multiproc,
            nevt=nevt,
            ioff=ioff,
            chunk=ds.config["chunksize"])


if __name__ == "__main__":
    main(sys.argv[1:])
