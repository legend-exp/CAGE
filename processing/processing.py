#!/usr/bin/env python3
import sys, os
import json
import numpy as np
import argparse
from parse import parse
import pandas as pd
import subprocess as sp
from pprint import pprint
from collections import OrderedDict
import tinydb as db
from tinydb.storages import MemoryStorage

# import matplotlib.pyplot as plt
# plt.style.use('../clint.mpl')

from pygama.flow import DataGroup

import pygama.math.histogram as pgh
from pygama.raw import build_raw
from pygama.dsp import build_dsp

# import lgdo.lh5_store as lh5 # pygama > 1.1.0
import pygama.lgdo.lh5_store as lh5 # pygama 1.1.0


def main():
    doc="""
    CAGE data processing routine.
    """
    rthf = argparse.RawTextHelpFormatter
    par = argparse.ArgumentParser(description=doc, formatter_class=rthf)
    arg, st, sf = par.add_argument, 'store_true', 'store_false'

    # declare group of files of interest.  supports sql-style(ish) queries
    arg('-q', '--query', nargs=1, type=str,
        help="select file group to calibrate: -q 'run==1 and [condition]' ")

    # routines
    arg('--d2r', action=st, help='run daq_to_raw')
    arg('--r2d', action=st, help='run raw_to_dsp')
    arg('--r2d_file', nargs=2, type=str, help='single-file raw_to_dsp')
    arg('--d2h', action=st, help='run dsp_to_hit (CAGE-specific)')

    # options
    arg('-o', '--over', action=st, help='overwrite existing files')
    arg('-n', '--nwfs', nargs='*', type=int, help='limit num. waveforms')
    arg('-v', '--verbose', action=st, help='verbose mode')
    arg('--mc', action=st, help='process with multi-channel mode (n+ readout)')

    arg('--dsp', nargs=1, type=str, help='name of dsp config file')
    arg('-u', '--user', action=st, help='user lh5 mode')
    arg('-s', '--spec', nargs=1, type=int, help='select alt set of calib peaks')
    arg('-l', '--lowE', action=st, help='calibrate low-E region with different calibration constants')
    arg('--epar', nargs=1, type=str,
        help="specify raw energy parameters to calibrate: --epar 'asd sdf dfg' ")

    args = par.parse_args()

    # -- load inputs --
    dg = DataGroup(os.path.expandvars('$CAGE_SW/processing/cage.json'), load=True)
    if args.query:
        que = args.query[0]
        dg.fileDB.query(que, inplace=True)
    else:
        dg.fileDB = dg.fileDB[-1:]

    # set additional options
    nwfs = args.nwfs[0] if args.nwfs is not None else np.inf
    if args.epar: dg.config['rawe'] = args.epar[0].split(' ')

    # -- show status before processing --
    print('Processing settings:'
          f'\n  overwrite? {args.over}'
          f'\n  limit wfs? {nwfs}'
          f'\n  multichannel d2r? {args.mc}')

    for envar in ['CAGE_SW','CAGE_DAQ','CAGE_LH5','CAGE_LH5_USER']:
        print(f'  {envar}', os.environ.get(envar))

    print(f'Current data group: {len(dg.fileDB)} files.')
    view_cols = ['run','cycle','daq_file','runtype']
    print(dg.fileDB[view_cols])

    # -- run routines --
    if args.d2r: d2r(dg, args.over, nwfs, args.verbose, args.user, args.mc)
    if args.r2d: r2d(dg, args.over, nwfs, args.verbose, args.user, args.mc, args.dsp[0] if args.dsp is not None else None)
    if args.d2h: d2h(dg, args.over, nwfs, args.verbose, args.user, args.lowE)

    if args.r2d_file:
        f_raw, f_dsp = args.r2d_file
        r2d_file(f_raw, f_dsp, args.over, nwfs, args.verbose, args.user)


def d2r(dg, overwrite=False, nwfs=None, verbose=False, user=False, run_mc=False):
    """
    $ ./processing.py -q 'run==[something]' --d2r
    run daq_to_raw on the current DataGroup
    """
    # print(dg.fileDB)
    # print(dg.fileDB.columns)

    subs = dg.subsystems # can be blank: ['']
    # subs = ['geds'] # TODO: ignore other datastreams
    # chans = ['g035', 'g042'] # TODO: select a subset of detectors

    print(f'Processing {dg.fileDB.shape[0]} files ...')

    for i, row in dg.fileDB.iterrows():

        lh5_dir = dg.lh5_user_dir if user else dg.lh5_dir

        f_daq = f"{dg.daq_dir}/{row['daq_dir']}/{row['daq_file']}"
        f_raw = f"{lh5_dir}/{row['raw_path']}/{row['raw_file']}"
        # f_raw = 'test.lh5'
        subrun = row['cycle'] if 'cycle' in row else None

        if pd.isna(row['raw_path']) or pd.isna(row['raw_file']):
            print('Error, nans in filenames/paths.  Check your fileDB.')
            print('f_daq:', f_daq)
            print('f_raw:', f_raw)
            print(row)
            exit()

        if not overwrite and os.path.exists(f_raw):
            print('file exists, overwrite not set, skipping f_raw:\n   ', f_raw)
            continue

        cyc = row['cycle']
        if row.skip:
            print(f'Cycle {cyc} has been marked junk, will not process.')
            continue

        if run_mc:
            # see pygama/io/ch_group.py for examples
            dg.config['ch_groups'] = {
                "ORSIS3302DecoderForEnergy" : {
                    "ch{ch:0>1d}" : {
                        # "ch_list" : [[144,151]], # check get_ccc.  range should work but doesn't.
                        "ch_list" : [146, 150], # channels 2 and 6 on the struck card
                        "system" : ""
                    }
                }
            }

        print(f'Processing cycle {cyc}')
        build_raw(in_stream=f_daq, in_stream_type='ORCA', out_spec='metadata/orca_config.json', verbose=verbose, n_max=nwfs, overwrite=overwrite, f_raw=f_raw)

def r2d(dg, overwrite=False, nwfs=None, verbose=False, user=False, run_mc=False, dsp=None):
    """
    $ ./processing.py -q 'run==[something]' --r2d
    """
    # load default DSP config file
    dsp_dir = os.path.expandvars('$CAGE_SW/processing/metadata/dsp/')
    
    # f_config = f'{dsp_dir}/dsp_07.json'
    # print('WARNING: clint hardcoded the DSP file for now')
    
    if dsp is None:
        with open(dsp_dir + 'dsp_07.json') as f:
            f_config = json.load(f, object_pairs_hook=OrderedDict)
    else:
        with open(dsp_dir + dsp) as f:
            f_config = json.load(f, object_pairs_hook=OrderedDict)
            
    print('using DSP file:', f_config)

    for i, row in dg.fileDB.iterrows():
        lh5_dir = dg.lh5_user_dir if user else dg.lh5_dir

        f_raw = f"{dg.lh5_dir}/{row['raw_path']}/{row['raw_file']}"
        f_dsp = f"{lh5_dir}/{row['dsp_path']}/{row['dsp_file']}"
        print(f_dsp)

        if "sysn" in f_raw:
            tmp = {'sysn' : 'geds'} # hack for lpgta
            f_raw = f_raw.format_map(tmp)
            f_dsp = f_dsp.format_map(tmp)

        if not overwrite and os.path.exists(f_dsp):
            print('file exists, overwrite not set, skipping f_dsp:\n   ', f_dsp)
            continue

        cyc = row['cycle']
        if row.skip:
            print(f'Cycle {cyc} has been marked junk, will not process.')
            continue

        # # load updated dsp config file
        # if row.dsp_id > 0 and dsp is None:
        #     f_config = dsp_dir + f'dsp_{row.dsp_id:02d}.json'
        #     print(f'Using DSP config: {f_config}')
            
        # load 2-channel DSP configs.  this is kinda hacky but should work
        if run_mc:
            lh5_tables = ['ch146/raw', 'ch150/raw']
            chan_config = {'ch146/raw':f_config,
                           'ch150/raw':f'{dsp_dir}/dsp/dsp_nplus.json'}
        else:
            lh5_tables, chan_config = None, None

        # NOTE: there is currently no smart DSP DB lookup here,
        # so the "db defaults" values in each of the JSON files will be used.

        print(f'Processing cycle {cyc}')
        build_dsp(f_raw, f_dsp, f_config, n_max=nwfs, write_mode='r')#, chan_config=chan_config)


def r2d_file(f_raw, f_dsp, overwrite=True, nwfs=None, verbose=False):
    """
    $ ./processing.py -q 'run==[something]' --r2d_file
    single-file mode, for testing
    """
    print('raw_to_dsp, single-file mode.')
    print('  input:', f_raw)
    print('  output:', f_dsp)

    # always overwrite
    if os.path.exists(f_dsp):
        os.remove(f_dsp)

    with open('oppi_dsp.json') as f:
        dsp_config = json.load(f, object_pairs_hook=OrderedDict)

    raw_to_dsp(f_raw, f_dsp, dsp_config, n_max=nwfs, verbose=verbose,
               overwrite=overwrite)


def d2h(dg, overwrite=False, nwfs=None, verbose=False, user=False, lowE=False):
    """
    $ ./processing.py -q 'run==[something]' --d2h

    Create a pygama 'hit' file with calibrated energy values.
    This routine assumes you can always calculate the calibration curve from
    the run you're considering.  For a test stand, this should work "Okay".

    IMPORTANT: by convention the ecalDB lookups are by 'run, cyclo, cychi'

    FUTURE TODO: It is pretty important to be able to apply a set of calibration
    constants from one group of cycles to another.   This could involve making
    more than one query to fileDB, which this CAGE routine won't support for now.
    To do this well (and cover many use cases), it might be better to make
    dsp_to_hit its own script with more flexibility in implementing this.  To me
    it seems like the best way to specify 'which calibration for which file' is
    to encode the correct input to use as a column of fileDB, probably computed
    based on cycle number or unix timestamp, and have methods for updating /
    modifying it.
    """
    # load ecal config file
    f_ecal = dg.config['ecal_default']
    if 'spec_id' in dg.config:
        spec_id = dg.config['spec_id'][0]
        print(f'spec_id: {spec_id}')
        if spec_id == 1:
            f_ecal = './metadata/config_ecal_ba.json'
            print(f'Loading Ba133 calibration parameters from: {f_ecal}')
    if lowE:
        f_ecal = './metadata/config_ecal_60keV.json'
        print(f'Loading 60 keV 241Am calibration parameters from: {f_ecal}')
    else:
        print(f'Loading default calibration parameters:\n  {f_ecal}')

    # merge main and ecal config dicts
    with open(os.path.expandvars(f_ecal)) as f:
        config = {**dg.config, **json.load(f)}
    dg.config = config

    # set additional options
    if 'rawe' not in dg.config:
        dg.config['rawe'] = dg.config['rawe_default']

    # dg.config['dsp_input_dir'] = dg.lh5_dir # using dsp files in CAGE LH5 directory
    dg.config['dsp_input_dir'] = dg.lh5_user_dir if user else dg.lh5_dir # comment in if using dsp files in user directory
    print('input DSP file dir: ', dg.config['dsp_input_dir'])

    dg.config['hit_output_dir'] = dg.lh5_user_dir if user else dg.lh5_dir

    print('  Energy parameters to calibrate:', dg.config['rawe'])

    # run dsp_to_hit (the pandas apply is more useful than a regular loop)
    dg.fileDB.apply(dsp_to_hit, axis=1, args=(dg, verbose, overwrite, lowE))


def dsp_to_hit(df_row, dg=None, verbose=False, overwrite=False, lowE=False):
    """
    Create hit files from dsp files.  This routine is specific to CAGE but could
    be extended & modified in the future to work for multi-channel data (PGT,
    L200, etc.)
    """
    apply_ecal = True
    apply_tscorr = False # not needed, should be fixed by the jan 30 2021 re-d2r

    f_dsp = f"{dg.config['dsp_input_dir']}/{df_row['dsp_path']}/{df_row['dsp_file']}"
    f_hit = f"{dg.config['hit_output_dir']}/{df_row['hit_path']}/{df_row['hit_file']}"
    # change output directory if in spec_id 2 mode (ie low-energy calibration to get 60 keV in right place)
    if lowE:
        f_hit = f"{dg.config['hit_output_dir']}/{df_row['hit_path']}/lowE/{df_row['hit_file']}"
        print(f'Writing to low-energy hit file: {f_hit}')
    if verbose:
        print('input:', f_dsp)
        print('output:', f_hit)

    if not overwrite and os.path.exists(f_hit):
        print('file exists, overwrite not set, skipping f_hit:\n   ', f_dsp)
        return

    # get run and cycle for ecalDB lookup.  also apply run selection
    run, cycle = df_row[['run', 'cycle']].astype(int)
    if df_row.skip:
        print(f'Cycle {cycle} has been marked junk, will not process.')
        return

    # create initial 'hit' DataFrame from dsp data
    hit_store = lh5.LH5Store()
    data, n_rows = hit_store.read_object(dg.config['input_table'], f_dsp)
    df_hit = data.get_dataframe()

    # 1. get energy calibration for this run from peakfit
    if apply_ecal:

        # loading the tinydb this way preserves the in-file text formatting
        cal_db = db.TinyDB(storage=MemoryStorage)
        with open(dg.config['ecaldb']) as f:
            raw_db = json.load(f)
            cal_db.storage.write(raw_db)

        # loop over energy estimators of interest
        for etype in dg.config['rawe']:

            # load ecalDB table
            tb = cal_db.table(f'peakfit_{etype}').all()
            df_cal = pd.DataFrame(tb)
            for col in ['run','cyclo','cychi']:
                df_cal[col] = df_cal[col].astype(int)

            # load cal constants for this cycle
            que = f'run=={run} and cyclo <= {cycle} <= cychi'
            df_run = df_cal.query(que)
            if len(df_run) != 1:
                print('Warning, non-unique query:', que)
                print(df_run)
                exit()

            # figure out the order of the polynomial from column names
            pols = {}
            for col in [c for c in df_run.columns if 'cal' in c]:
                val = parse('cal{p}', col)
                val = val.named # convert to dict
                iord = int(val['p'])
                pols[iord] = df_run.iloc[0][f'cal{iord}']

            # get the coefficients in descending order for np.poly1d: p2, p1, p0...
            coeffs = []
            for ord, val in sorted(pols.items()):
                coeffs.append([ord, val])
            coeffs = np.array(coeffs)
            coeffs = coeffs[coeffs[:,0].argsort()[::-1]] # 2, 1, 0 ...
            coeffs = coeffs[:,1]

            # apply the calibration to the dataframe
            pfunc = np.poly1d(coeffs)
            df_hit[f'{etype}_cal'] = pfunc(df_hit[f'{etype}'])


    # 2. compute timestamp rollover correction (specific to struck 3302)
    clock = 100e6 # 100 MHz
    if apply_tscorr:
        UINT_MAX = 4294967295 # (0xffffffff)
        t_max = UINT_MAX / clock
        ts = df_hit['timestamp'].values / clock
        tdiff = np.diff(ts)
        tdiff = np.insert(tdiff, 0 , 0)
        iwrap = np.where(tdiff < 0)
        iloop = np.append(iwrap[0], len(ts))
        ts_new, t_roll = [], 0
        for i, idx in enumerate(iloop):
            ilo = 0 if i==0 else iwrap[0][i-1]
            ihi = idx
            ts_block = ts[ilo:ihi]
            t_last = ts[ilo-1]
            t_diff = t_max - t_last
            ts_new.append(ts_block + t_roll)
            t_roll += t_last + t_diff
        df_hit['ts_sec'] = np.concatenate(ts_new)
    else:
        # NOTE: may need to subtract off the 1st value here if we find
        # that the timestamp doesn't reset at cycle boundaries.
        df_hit['ts_sec'] = df_hit['timestamp'].values / clock


    # 3. compute global timestamp
    t_start = df_row['startTime']
    if t_start is not None:
        df_hit['ts_glo'] = df_hit['ts_sec'] + t_start

    # write to LH5 file
    if os.path.exists(f_hit):
        os.remove(f_hit)
    sto = lh5.Store()
    tb_name = dg.config['input_table'].replace('dsp', 'hit')
    tb_lh5 = lh5.Table(size=len(df_hit))

    for col in df_hit.columns:
        tb_lh5.add_field(col, lh5.Array(df_hit[col].values, attrs={'units':''}))
        if verbose:
            print(col)

    print(f'Writing table: {tb_name} in file:\n   {f_hit}')
    sto.write_object(tb_lh5, tb_name, f_hit)

    if verbose:
        print('Creating diagnostic plots ...')

        # energy
        xlo, xhi, xpb = 0, 3000, 10
        hist, bins, _ = pgh.get_hist(df_hit['trapEftp_cal'], range=(xlo, xhi), dx=xpb)
        plt.semilogy(bins[1:], hist, ds='steps', c='b', lw=1)
        plt.xlabel('Energy (keV)', ha='right', x=1)
        plt.ylabel('Counts', ha='right', y=1)
        plt.savefig('./plots/d2h_etest.png')
        print('saved figure: ./plots/d2h_etest.png')
        plt.cla()

        # timestamp
        xv = np.arange(len(df_hit))
        plt.plot(xv, df_hit['ts_sec'], '.b')
        plt.savefig('./plots/d2h_ttest.png')
        print('saved figure: ./plots/d2h_ttest.png')
        plt.cla()

        # exit, don't create + overwrite a million plots
        print('verbose mode of d2h is meant to look at 1 cycle file, exiting...')
        exit()


if __name__=="__main__":
    main()
