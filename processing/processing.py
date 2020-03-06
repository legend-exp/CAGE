#!/usr/bin/env python3
import sys, os, io
import json
import argparse
import numpy as np
import pandas as pd
from pprint import pprint
from pygama import DataSet
import pygama.utils as pu

def main(argv):
    """
    Uses pygama's amazing DataSet class to process runs
    for different data sets and arbitrary configuration options
    defined in a JSON file.
    """
    # datadir = os.environ["CAGEDATA"]
    run_db, cal_db = f'./runDB.json', f'./calDB.json'

    # -- parse args --
    par = argparse.ArgumentParser(description="data processing suite for MJ60")
    arg, st, sf = par.add_argument, "store_true", "store_false"
    arg("-ds", nargs='*', action="store", help="load runs for a DS")
    arg("-r", "--run", nargs=1, help="load a single run")
    arg("-d2r", "--daq_to_raw", action=st, help="run daq_to_raw on list")
    arg("-r2d", "--raw_to_dsp", action=st, help="run raw_to_dsp on list")
    arg("-t", "--test", action=st, help="test mode, don't run")
    arg("-n", "--nevt", nargs='?', default=np.inf, help="limit max num events")
    arg("-i", "--ioff", nargs='?', default=0, help="start at index [i]")
    arg("-v", "--verbose", action=st, help="set verbose output")
    arg("-o", "--ovr", action=st, help="overwrite existing files")
    arg("-m", "--nomp", action=sf, help="don't use multiprocessing")
    args = vars(par.parse_args())

    # -- declare the DataSet --
    ds = pu.get_dataset_from_cmdline(args, run_db, cal_db)

    # print(ds.runs)
    # pprint(ds.paths)

    # -- start processing --
    if args["daq_to_raw"]:
        daq_to_raw(ds, args["ovr"], args["nevt"], args["verbose"], args["test"])

    if args["raw_to_dsp"]:
        raw_to_dsp(ds, args["ovr"], args["nevt"], args["ioff"], args["nomp"],
                   args["verbose"], args["test"])


def daq_to_raw(ds, overwrite=False, nevt=np.inf, v=False, test=False):
    """
    Run daq_to_raw on a set of runs.
    [raw file] ---> [raw_run{}.lh5] (basic info & waveforms)
    """
    from pygama.io.daq_to_raw import daq_to_raw

    for run in ds.runs:

        daq_file = ds.paths[run]["daq_path"]
        raw_file = ds.paths[run]["raw_path"]
        if raw_file is not None and overwrite is False:
            print("file exists, overwrite flag isn't set.  continuing ...")
            continue

        conf = ds.paths[run]["build_opt"]
        opts = ds.config["build_options"][conf]["daq_to_raw_options"]

        if test:
            print("test mode (dry run), processing DAQ file:", daq_file)
            print("output file:", raw_file)
            continue

        # # old pandas version
        # daq_to_raw(daq_file, run, verbose=v, output_dir=ds.raw_dir,
        #              overwrite=overwrite, n_max=nevt, config=ds.config)#settings=opts)

        # new lh5 version
        daq_to_raw(daq_file, raw_filename=raw_file, run=run, chan_list=None,
                   n_max=nevt, verbose=False, output_dir=ds.raw_dir,
                   overwrite=overwrite, config=ds.config)


def raw_to_dsp(ds, overwrite=False, nevt=None, ioff=None, multiproc=True,
               verbose=False, test=False):
    """
    Run raw_to_dsp on a set of runs.
    [raw_run{}.lh5] ---> [dsp_run{}.lh5]  (tier 2 file: DSP results, no waveforms)

    Can declare the processor list via:
    - json configuration file (recommended)
    - Intercom(default_list=True)
    - manually add with Intercom::add
    """
    from pygama.dsp.dsp_base import Intercom
    from pygama.io.raw_to_dsp import RunDSP

    for run in ds.runs:

        raw_file = ds.paths[run]["raw_path"]
        dsp_file = ds.paths[run]["dsp_path"]
        if dsp_file is not None and overwrite is False:
            continue

        if test:
            print("test mode (dry run), processing raw file:", raw_file)
            continue

        conf = ds.paths[run]["build_opt"]
        proc_list = ds.config["build_options"][conf]["raw_to_dsp_options"]
        proc = Intercom(proc_list)

        RunDSP(
            raw_file,
            proc,
            output_dir=ds.dsp_dir,
            overwrite=overwrite,
            verbose=verbose,
            multiprocess=multiproc,
            nevt=nevt,
            ioff=ioff,
            chunk=ds.config["chunksize"])


if __name__ == "__main__":
    main(sys.argv[1:])
