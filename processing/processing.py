#!/usr/bin/env python3
import sys, os, io
import json
import argparse
import numpy as np
import pandas as pd
from pprint import pprint
from pygama import DataSet
import pygama.utils as pu
from pygama.dsp.ProcessingChain import ProcessingChain
from pygama.dsp.transforms import *
from pygama.dsp.units import *
from pygama.io import io_base as io


def main(argv):
    """
    Uses pygama's amazing DataSet class to process runs
    for different data sets and arbitrary configuration options
    defined in a JSON file.
    """
    # datadir = os.environ["CAGEDATA"]
    run_db, cal_db = f'./runDB.json', f'./calDB.json'

    # -- parse args --
    par = argparse.ArgumentParser(description="data processing suite for CAGE")
    arg, st, sf = par.add_argument, "store_true", "store_false"
    arg("-ds", nargs='*', action="store", help="load runs for a DS")
    arg("-r", "--run", nargs=1, help="load a single run")
    arg("-d2r", "--daq_to_raw", action=st, help="run daq_to_raw on list")
    arg("-r2d", "--raw_to_dsp", action=st, help="run raw_to_dsp on list")
    arg("-t", "--test", action=st, help="test mode, don't run")
    arg("-n", "--nevt", nargs='?', default=np.inf, help="limit max num events")
    arg("-i", "--ioff", nargs='?', default=0, help="start at index [i]")
    arg("-o", "--ovr", action=st, help="overwrite existing files")

    arg('-v', '--verbose', default=2, type=int,
        help="Verbosity level: 0=silent, 1=basic warnings, 2=verbose output, 3=debug. Default is 2.")

    arg('-b', '--block', default=8, type=int,
                        help="Number of waveforms to process simultaneously. Default is 8")

    arg('-g', '--group', default='/daqdata',
                        help="Name of group in LH5 file. Default is daqdata.")

    # -- declare the DataSet --
    args = par.parse_args()
    d_args = vars(par.parse_args())
    ds = pu.get_dataset_from_cmdline(d_args, run_db, cal_db)

    # print(ds.runs)
    # pprint(ds.paths)

    # -- start processing --
    if args.daq_to_raw:
        daq_to_raw(ds, args.ovr, args.nevt, args.verbose, args.test)

    if args.raw_to_dsp:
        raw_to_dsp(ds, args.ovr, args.nevt, args.test, args.verbose, args.block,
                   args.group)


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
                   prefix=ds.rawpre, n_max=nevt, verbose=False, output_dir=ds.raw_dir,
                   overwrite=overwrite, config=ds.config)


def raw_to_dsp(ds, overwrite=False, nevt=None, test=False, verbose=2, block=8,
               group='daqdata'):
    """
    Run raw_to_dsp on a set of runs.
    [raw file] ---> [dsp_run{}.lh5] (digital signal processing results)
    """
    for run in ds.runs:
        raw_file = ds.paths[run]["raw_path"]
        dsp_file = ds.paths[run]["dsp_path"]

        if dsp_file is not None and overwrite is False:
            continue

        if dsp_file is None:
            # declare new file name
            dsp_file = raw_file.replace('raw_', 'dsp_')

        if test:
            print("test mode (dry run), processing raw file:", raw_file)
            continue

        # new LH5 version
        lh5 = io.LH5Store()
        data = lh5.read_object("/ORSIS3302DecoderForEnergy", raw_file)

        wf_in = data['waveform']['values'].nda
        dt = data['waveform']['dt'].nda[0] * unit_parser.parse_unit(data['waveform']['dt'].attrs['units'])

        # Set up processing chain
        proc = ProcessingChain(block_width=block, clock_unit=dt, verbosity=verbose)
        proc.add_input_buffer("wf", wf_in, dtype='float32')

        proc.add_processor(mean_stdev, "wf[0:1000]", "bl", "bl_sig")
        proc.add_processor(np.subtract, "wf", "bl", "wf_blsub")
        proc.add_processor(pole_zero, "wf_blsub", 70*us, "wf_pz")
        proc.add_processor(trap_filter, "wf_pz", 10*us, 5*us, "wf_trap")
        proc.add_processor(np.amax, "wf_trap", 1, "trapmax", signature='(n),()->()', types=['fi->f'])
        proc.add_processor(np.divide, "trapmax", 10*us, "trapE")
        proc.add_processor(avg_current, "wf_pz", 10, "curr")
        proc.add_processor(np.amax, "curr", 1, "A_10", signature='(n),()->()', types=['fi->f'])
        proc.add_processor(np.divide, "A_10", "trapE", "AoE")

        # Set up the LH5 output
        lh5_out = io.LH5Table(size=proc.__buffer_len__)
        lh5_out.add_field("trapE", io.LH5Array(proc.get_output_buffer("trapE"),
                                               attrs={"units":"ADC"}))
        lh5_out.add_field("bl", io.LH5Array(proc.get_output_buffer("bl"),
                                            attrs={"units":"ADC"}))
        lh5_out.add_field("bl_sig", io.LH5Array(proc.get_output_buffer("bl_sig"),
                                                attrs={"units":"ADC"}))
        lh5_out.add_field("A", io.LH5Array(proc.get_output_buffer("A_10"),
                                           attrs={"units":"ADC"}))
        lh5_out.add_field("AoE", io.LH5Array(proc.get_output_buffer("AoE"),
                                             attrs={"units":"ADC"}))

        print("Processing:\n",proc)
        proc.execute()

        print("Writing to: ", dsp_file)
        lh5.write_object(lh5_out, "data", dsp_file)




if __name__ == "__main__":
    main(sys.argv[1:])
