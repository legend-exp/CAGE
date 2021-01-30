#!/usr/bin/env python3
import os
import re
import time
import json
import argparse
import pandas as pd
import numpy as np
import h5py
from pprint import pprint
from datetime import datetime
import itertools
from scipy.optimize import curve_fit
import tinydb as db
from tinydb.storages import MemoryStorage
import matplotlib
if os.environ.get('HOSTNAME'): # cenpa-rocks
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('../clint.mpl')
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from tqdm import tqdm
    tqdm.pandas() 

from pygama import DataGroup
from pygama.io.orcadaq import parse_header
import pygama.lh5 as lh5
import pygama.analysis.metadata as pmd
import pygama.analysis.histograms as pgh
import pygama.analysis.calibration as pgc
import pygama.analysis.peak_fitting as pgf

def main():
    doc="""
    Energy calibration app for CAGE.

    Usage:
    First, generate an up-to-date fileDB (setup.py) and DSP files (processing.py).
      You will need to run setup.py with --orca and --rt options.
    
    Select a group of files to calibrate with a query:
        $ ./energy_cal.py -q 'run==234 [and cycle <= 345, etc.]'
    
    Check the raw spectrum with '--raw' (default estimator: trapEmax)
    
    Adjust the JSON configuration file as needed (default: config_ecal.json)
    
    Run "peakdet", which calculates up to 2nd-order calibration constants for
      each channel, y = p0 +  p1 * x  +  p2 * x**2, and saves them as tables
      in our ecalDB file.
    
    Run "peakfit", which fits each peak of interest to a peakshape function 
      (default: gaussian + linear step function), computes calibration 
      constants, and resolution curves, and saves results to ecalDB.

    Results are saved ('-w' option) to JSON format with 'legend-metadata' 
      style conventions.
    
    -- C. Wiseman, T. Mathew, G. Othman, J. Detwiler
    """
    rthf = argparse.RawTextHelpFormatter
    par = argparse.ArgumentParser(description=doc, formatter_class=rthf)
    arg, st, sf = par.add_argument, 'store_true', 'store_false'

    # declare group of files of interest.  supports sql-style(ish) queries
    arg('-q', '--query', nargs=1, type=str,
        help="select file group to calibrate: -q 'run==1 and [condition]' ")
    
    # primary ops
    arg('--raw', action=st, help='display/save uncalibrated energy histogram')
    arg('-pd', '--peakdet', action=st, help='first pass: peak detection')
    arg('-pi', '--peakinp', nargs=1, type=str, help='first pass: manually input peaks')
    arg('-pf', '--peakfit', action=st, help='second pass: individual peak fit')
    arg('--all', action=st, help='run all passes, write to db')

    # options
    arg('-v', '--verbose', nargs=1, help='set verbosity (default: 1)')
    arg('--init_db', action=st, help='initialize ecal database JSON file')
    arg('-u', '--lh5_user', action=st, help='user lh5 mode')
    arg('-w', '--write_db', action=st, help='write results to ecalDB file')
    arg('-s', '--show_db', nargs='*', help='show ecalDB, optionally specify table name')
    arg('-p', '--show_plot', action=st, help='show debug plot')
    arg('-b', '--batch', action=st, help="batch mode: save & don't display plots")
    arg('--show_config', action=st, help='show current configuration')
    arg('--match', nargs=1, type=str, help='set peak match mode (default: first)')
    arg('--pol', nargs=1, type=int, help='set peakdet/peakinput pol order')
    arg('--epar', nargs=1, type=str,
        help="specify raw energy parameters: --epar 'asd sdf dfg' ")
    arg('-gb', '--group', nargs=1, type=str,
        help="select alternate groupby: -gb 'run cycle' ")
    arg('-ff', '--fit_func', nargs=1, type=str, help='set peakfit fit function (default is gaus+step)')
    arg('--spec', nargs=1, type=int, help='select alternate set of peaks to calibrate')
    args = par.parse_args()


    # -- set up fileDB and config dictionary --
    
    # query the fileDB & modify in-memory to only contain files matching our query
    dg = DataGroup('cage.json', load=True)
    if args.query:
        que = args.query[0]
        dg.fileDB.query(que, inplace=True)
    else: 
        dg.fileDB = dg.fileDB[-1:]

    # load ecal config file
    f_ecal = dg.config['ecal_default']
    if args.spec:
        spec_id = args.spec[0]
        if spec_id == 1:
            f_ecal = './metadata/config_ecal_ba.json' 
            print(f'Loading Ba133 calibration parameters from: {f_ecal}')
        else:
            print('Error, unknown calib mode:', args.spec[0])
    else:
        print(f'Loading default calibration parameters from: {f_ecal}')
        
    # merge main and ecal config dicts
    with open(os.path.expandvars(f_ecal)) as f:
        config = {**dg.config, **json.load(f)}
    
    # initialize ecalDB JSON output file.  only run this once
    if args.init_db:
        init_ecaldb(config)
        exit()
    try:
        # load ecalDB this way (into memory) s/t the pretty on-disk formatting isn't changed
        db_ecal = db.TinyDB(storage=MemoryStorage)
        with open(config['ecaldb']) as f:
            raw_db = json.load(f)
            db_ecal.storage.write(raw_db)
    except:
        print('JSON database file not found or corrupted.  Rerun --init_db')
        exit()

    # set more options -- everything should be loaded into the 'config' dict
    config['gb_cols'] = args.group[0].split(' ') if args.group else ['run']
    if config['gb_cols'][0] != 'run':
        print("Error, first groupby column must be 'run'!  Try -gb 'run cycle'")
        exit() 

    # set input data directory (CAGE_LH5, CAGE_LH5_USER, or cwd)
    lh5_dir = dg.lh5_user_dir if args.lh5_user else dg.lh5_dir
    config['lh5_dir'] = os.path.expandvars(lh5_dir)
    config['pol'] = args.pol if args.pol else [2]
    config['rawe'] = args.epar[0].split(' ') if args.epar else config['rawe_default']
    config['match_mode'] = args.match if args.match else 'first'
    config['mp_tol'] = 100 # raw peaks must be within keV
    config['batch_mode'] = True if args.batch else False
    config['show_plot'] = True if args.show_plot else False
    config['write_db'] = True if args.write_db else False
    if args.peakinp: config['input_id'] = args.peakinp[0]
    config['input_peaks'] = './metadata/input_peaks.json'
    config['fit_func'] = args.fit_func[0] if args.fit_func else 'gauss_step'
    config['verbose'] = args.verbose[0] if args.verbose else 0
    
    # include fields from ecalDB in the config dict
    config = {**config, **db_ecal.table('_file_info').all()[0]}


    # -- show status -- 
    
    ecal_cols = ['run', 'cycle', 'daq_file', 'runtype', 'startTime', 'threshold',
                 'stopTime', 'runtime']
    
    if dg.fileDB is None:
        print('Warning, no fileDB is loaded.')
    
    elif not all(x in dg.fileDB.columns for x in ecal_cols):
        print('Error, fileDB is missing some columns.  Did you run setup.py?')
        print('Current available columns:\n', dg.fileDB.columns)
        exit()
    
    print(f'Ready to calibrate.\n'
          f"Output file: {config['ecaldb']} \n"
          'Calibrating raw energy parameters:', config['rawe'], '\n'
          f'Current data group ({len(dg.fileDB)} files) --->> ')
    print(dg.fileDB[ecal_cols], '\n')
    
    if args.show_config:
        print('Current energy_cal config:')
        pprint(config)
        print('\n')
        
    if args.show_db is not None:
        show_ecaldb(config, args.show_db)


    # -- main routines --
    
    if args.raw:
        check_raw_spectrum(dg, config, db_ecal)
    
    if args.peakdet: 
        run_peakdet(dg, config, db_ecal)
    
    if args.peakfit: 
        run_peakfit(dg, config, db_ecal)

    if args.all:
        config['write_db'] = True
        run_peakdet(dg, config, db_ecal)
        run_peakfit(dg, config, db_ecal)


def init_ecaldb(config):
    """
    $ ./energy_cal.py --init_db
    one-time set up of primary database file.
    You probably DON'T want to run this, it will wipe ecalDB.json
    """
    ans = input('(Re)create main ecal JSON file?  Are you really sure? (y/n) ')
    if ans.lower() != 'y':
        exit()

    f_db = config['ecaldb'] # for pgt, should have one for each detector

    if os.path.exists(f_db):
        os.remove(f_db)

    # create the database in-memory
    db_ecal = db.TinyDB(storage=MemoryStorage)
    query = db.Query()

    # create a table with metadata (provenance) about this calibration file
    file_info = {
        "system" : config['system'],
        "cal_type" : "energy",
        "created_gmt" : datetime.utcnow().strftime("%m/%d/%Y, %H:%M:%S"),
        "input_table" : config['input_table']
        }
    db_ecal.table('_file_info').insert(file_info)

    # pretty-print the JSON database to file
    raw_db = db_ecal.storage.read()
    pmd.write_pretty(raw_db, f_db)

    # show the file as-is on disk
    with open(f_db) as f:
        print(f.read())


def show_ecaldb(config, tables=None):
    """
    $ ./energy_cal.py --show_db [table name]
    """
    print('Loading ecalDB ...')
    
    # # show the file as-is on disk
    # with open(config['ecaldb']) as f:
    #     print(f.read())

    # make sure the file is usable by TinyDB
    db_ecal = db.TinyDB(storage=MemoryStorage)
    with open(config['ecaldb']) as f:
        raw_db = json.load(f)
        db_ecal.storage.write(raw_db)

    # show tables in ecalDB, in pandas format.  user either passes
    # a specific table to look at, or we print them all.
    if tables is not None and len(tables)==0:
        print('No table name given, showing all available tables in ecalDB.')
        tables = [tb for tb in db_ecal.tables() if tb != '_file_info']
    else:
        tb_list = tables
    
    # show user requested tables
    for tb in tables:
        print('\necalDB table:', tb)
        db_table = db_ecal.table(tb).all()
        df_table = pd.DataFrame(db_table)
        
        # can't save ints correctly to tinyDB (yet), so fix them here
        int_cols = [col for col in ['run','cychi','cyclo','calpass'] if col in df_table.columns]
        for col in int_cols:
            df_table[col] = df_table[col].astype(int)
    
        # fix the column order too
        cols = ['run','cyclo','cychi']
        cols += [c for c in df_table.columns if c not in cols]
    
        print(df_table[cols])
    

def check_raw_spectrum(dg, config, db_ecal):
    """
    $ ./energy_cal.py -q 'query' --raw
    """
    # load energy data
    dsp_list = config['lh5_dir'] + dg.fileDB['dsp_path'] + '/' + dg.fileDB['dsp_file']
    raw_data = lh5.load_nda(dsp_list, config['rawe'], config['input_table'], verbose=False)
    runtime_min = dg.fileDB['runtime'].sum()

    print('\nShowing raw spectra ...')
    for etype in config['rawe']:
        xlo, xhi, xpb = config['init_vals'][etype]["raw_range"]

        # load energy data for this estimator
        data = raw_data[etype]

        # print columns of table
        file_info = db_ecal.table('_file_info').all()[0]
        tb_in = file_info['input_table']
        with h5py.File(dsp_list.iloc[0], 'r') as hf:
            print("LH5 columns:", list(hf[f'{tb_in}'].keys()))

        # generate histogram
        hist, bins, var = pgh.get_hist(data, range=(xlo, xhi), dx=xpb)
        bins = bins[1:] # trim zero bin, not needed with ds='steps'

        # normalize by runtime
        hist_rt = np.divide(hist, runtime_min * 60)

        print('\nPlease determine the following parameters for ecal config file:\n'
              "  - 'raw_range': Optimal binning, and hi/lo raw energy limits\n"
              "  - 'peakdet_thresh': ~1/2 the height of a target peak\n"
              "  - 'lowe_cut' energy threshold for peak detection")

        print(f'\nRaw E: {etype}, {len(data)} cts, runtime: {runtime_min:.2f} min')

        plt.semilogy(bins, hist_rt, ds='steps', c='b', lw=1, label=etype)
        plt.xlabel(etype, ha='right', x=1)
        plt.ylabel(f'cts/sec, {xpb}/bin', ha='right', y=1)

        if config['batch_mode']:
            plt.savefig('./plots/energy_cal/cal_spec_test.png')
        else:
            plt.show()
        plt.close()


def run_peakdet(dg, config, db_ecal):
    """
    $ ./energy_cal.py -q 'query' [-pd / -pi inp_id] [-p : show plot] [-w : write ecalDB]
    
    Run "first guess" calibration of a list of energy parameters.
    Creates a table in the ecalDB for each one, storing up to 2nd order
    polynomials: y = p0  +  p1 * x  +  p2 * x**2.
    These are used as inputs to "peakfit".

    We have two modes:
    -- automatic (default): find p1 by matching the ratios of uncalibrated
       auto-detected peaks to an input list of peaks in keV.  
       Assumes y = p1 * x, which may not always work for all detectors.
       
    -- "input peaks": use a JSON config file to set expected peak locations.
       This is useful when the spectrum deviates too much from y = p1 * x.
          
    Files are grouped by run, and optionally by cycle (calibrates individual files
      within the run.)  Right now, we require the first item in gb_cols to be 'run'.
      It's also possible to group a subset of files in a run together, with a 
      query like "run==123 and cycle > 456"

    We then write several TinyDB 'Tables' to our ecalDB file.  
    They have a nice 1--1 correspondence to pandas dataframes.
    """
    gb = dg.fileDB.groupby(config['gb_cols'])
    run = dg.fileDB.run.iloc[0]
    cyclo, cychi = dg.fileDB.cycle.iloc[0], dg.fileDB.cycle.iloc[-1]
    print(f'Running peakdet, run {run}, cycles {cyclo}--{cychi}')
    
    if 'input_id' in config.keys():
        pol = config['pol'][0]
        print(f'Fitting manually input peak locations to polynomial, order', pol)
        result = gb.apply(peakdet_input, *[config])
    else:
        print('Automatically detecting peaks based on input list.')
        result = gb.apply(peakdet_auto, *[config])
        
    # write separate tables for each energy estimator to the TinyDB
    for epar in config['rawe']:

        # format output table
        epar_cols = [r for r in result.columns if epar in r]
        df_epar = result[epar_cols].copy()
        df_epar.rename(columns={c:c.split('_')[-1] for c in epar_cols},
                       inplace=True)
        df_epar['calpass'] = df_epar['calpass'].astype(int)
        df_epar.reset_index(inplace=True)
        
        if 'cycle' in df_epar.columns:
            # this is redundant with cyclo and cychi
            df_epar.drop('cycle', 1, inplace=True)
        
        tb_name = f'peakinp_{epar}' if 'input_id' in config.keys() else f'peakdet_{epar}'
        print('Results:', tb_name)
        print(df_epar)
        
        # this is in-memory, no write to file yet
        table = db_ecal.table(tb_name)
        q = db.Query()
        for i, row in df_epar.iterrows():
            que = ((q.run==row.run) & (q.cyclo==row.cyclo) & (q.cychi==row.cychi))
            table.upsert(row.to_dict(), que)

    if not config['write_db']:
        print('Done. ecalDB write mode not set (-w option)')
        return

    # show in-memory state and then write to file
    # pprint(db_ecal.storage.read())
    print('Writing results to ecalDB.')
    pmd.write_pretty(db_ecal.storage.read(), config['ecaldb'])


def peakdet_auto(df_group, config):
    """
    Access all files in this group, load energy histograms, and find the
    "first guess" linear calibration constant.
    Return the value, and a bool indicating success.
    """
    # load data and compute runtime
    dsp_list = config['lh5_dir'] + df_group['dsp_path'] + '/' + df_group['dsp_file']
    edata = lh5.load_nda(dsp_list, config['rawe'], config['input_table'], verbose=False)
    runtime_min = df_group['runtime'].sum()
    run = df_group.run.iloc[0]
    cyclo, cychi = df_group.cycle.iloc[0], df_group.cycle.iloc[-1]
    print(f'  Runtime: {runtime_min:.1f} min.  Calibrating:', [f'{et}:{len(ev)} events' for et, ev in edata.items()])

    # loop over energy estimators of interest
    pd_results = {}
    for et in config['rawe']:

        # get histogram, error, normalize by runtime, and derivative
        xlo, xhi, xpb = config['init_vals'][et]['raw_range']
        hist, bins, var = pgh.get_hist(edata[et], range=(xlo, xhi), dx=xpb)
        hist_norm = np.divide(hist, runtime_min * 60)
        hist_err = np.array([np.sqrt(hbin / (runtime_min * 60)) for hbin in hist])

        # run peakdet
        pd_thresh = config['init_vals'][et]['peakdet_thresh']
        lowe_cut = config['init_vals'][et]['lowe_cut']
        ctr_bins = (bins[:-1] + bins[1:]) / 2.
        idx = np.where(ctr_bins > lowe_cut)

        maxes, mins = pgc.peakdet(hist_norm[idx], pd_thresh, ctr_bins[idx])
        # maxes, mins = pgc.peakdet(hist_deriv[idx], pd_thresh, ctr_bins[idx])
        if len(maxes)==0:
            print('warning, no maxima!  adjust peakdet threshold')
        # print(maxes) # x (energy) [:,0], y (counts) [:,1]

        # run peak matching
        exp_pks = config['expected_peaks']
        tst_pks = config['test_peaks']
        mode = config['match_mode']
        etol = config['raw_ene_tol']
        lin_cal, mp_success = match_peaks(maxes, exp_pks, tst_pks, mode, etol)
        
        if config['show_plot']:

            # plot uncalibrated and calibrated energy spectrum, w/ maxima
            fig, (p0, p1) = plt.subplots(2, 1, figsize=(8, 8))

            idx = np.where(bins[1:] > lowe_cut)
            imaxes = [np.where(np.isclose(ctr_bins, x[0]))[0][0] for x in maxes]
            imaxes = np.asarray(imaxes)

            # energy, uncalibrated
            p0.semilogy(bins[imaxes], hist_norm[imaxes], '.m')
            p0.semilogy(bins[idx], hist_norm[idx], ds='steps', c='b', lw=1, label=et)
            p0.set_ylabel(f'cts/s, {xpb}/bin', ha='right', y=1)
            p0.set_xlabel(et, ha='right', x=1)

            # energy, with rough calibration
            bins_cal = bins[1:] * lin_cal
            # p1.plot(bins_cal, hist_norm, ds='steps', c='b', lw=1,
            p1.semilogy(bins_cal, hist_norm, ds='steps', c='b', lw=1,
                    label=f'E = {lin_cal:.3f}*{et}')

            # compute best-guess location of all peaks, assuming rough calibration
            cal_maxes = lin_cal * maxes[:, 0]
            all_pks = np.concatenate((exp_pks, tst_pks))
            raw_guesses = []
            for pk in all_pks:

                imatch = np.isclose(cal_maxes, pk, atol=config['mp_tol'])
                if imatch.any():
                    print(pk, cal_maxes[imatch], maxes[:,0][imatch])
                    raw_guesses.append([pk, maxes[:,0][imatch][0]])

            if len(raw_guesses) != 0:
                rg = np.asarray(raw_guesses)
                rg = rg[rg[:,0].argsort()] # sort by energy
                cmap = plt.cm.get_cmap('jet', len(rg))
                for i, epk in enumerate(rg):
                    idx_nearest = (np.abs(bins_cal - epk[0])).argmin()
                    cts_nearest = hist_norm[idx_nearest]
                    p1.plot(epk[0], cts_nearest, '.r', c=cmap(i),
                            label=f'{epk[0]:.1f} keV')
                print('raw pk locations:\n', rg)

            p1.set_xlabel(f'{et}, pass-1 cal', ha='right', x=1)
            p1.set_ylabel(f'cts/s, {xpb} kev/bin', ha='right', y=1)
            p1.legend(fontsize=10)

            if config['batch_mode']:
                plt.savefig(f'./plots/energy_cal/peakdet_{et}_run{run}_clo{cyclo}_chi{cychi}.pdf')
            else:
                plt.show()

        pd_results[f'{et}_calpass'] = mp_success
        pd_results[f'{et}_runtime'] = runtime_min
        pd_results[f'{et}_pol0'] = 0 
        pd_results[f'{et}_pol1'] = lin_cal
        pd_results[f'{et}_cyclo'] = cyclo
        pd_results[f'{et}_cychi'] = cychi

    return pd.Series(pd_results)


def match_peaks(maxes, exp_pks, tst_pks, mode='first', ene_tol=10):
    """
    modes:
    - 'first' : pin the first expected peak, search for the first test peak
    - 'ratio' : compute ratio match
    """
    print('running autopeak matching.  mode is:', mode)
    
    if mode == 'first':

        # set expected and test peak
        exp_pk, tst_pk = exp_pks[0], tst_pks[0]
        print(f'Pinning {exp_pk} looking for {tst_pk}, tolerance: {ene_tol} keV')

        # loop over raw peaks, apply a linear cal, and see if there
        # is a raw peak near the test location, within an energy tolerance
        lin_cals = []
        for xpk in maxes[:,0]:
            lin_cal = exp_pk / xpk
            cal_maxes = lin_cal * maxes[:,0]
            imatch = np.isclose(cal_maxes, tst_pk, atol=ene_tol)
            if imatch.any():
                lin_cals.append(lin_cal)
        lin_cals = sorted(lin_cals)

        if len(lin_cals) == 0:
            print('Found no matches!')
            return 1, False
        elif len(lin_cals) > 1:
            print('Warning, found multiple matches. Using first one...')
            print(lin_cals)
            # exit()

        # first pass calibration constant
        return lin_cals[0], True

    elif mode == 'ratio':
        """
        # NOTE: maybe we can improve on "first" mode by computing all
        # permutations and returning a calibration constant that 'averages'
        # between the most correct ones.

        Uses a peak matching algorithm based on finding ratios of uncalibrated (u)
        and "true, keV-scale" (e) energies.
        We run peakdet to find the maxima in the spectrum, then compute all ratios:
            - e1/e2, u1/u2, ..., u29/u30 etc.
        We find the subset of uncalibrated ratios (u7/u8, ... etc) that match the
        "true" ratios, and compute a calibration constant for each.

        Then for each uncalibrated ratio, we assume it to be true, then loop over
        the expected peak positions.

        We shift the uncalibrated peaks so that the true peak would be very close
        to 0, and calculate its distance from 0.  The "true" calibration constant
        will minimize this value for all ratios, and this is the one we select.
        """

        # run peakdet to identify the uncalibrated maxima
        maxes, mins = pu.peakdet(h, pk_thresh, b)
        umaxes = np.array(sorted([x[0] for x in maxes], reverse=True))

        # compute all ratios
        ecom = [c for c in it.combinations(epeaks, 2)]
        ucom = [c for c in it.combinations(umaxes, 2)]
        eratios = np.array([x[0] / x[1] for x in ecom]) # assumes x[0] > x[1]
        uratios = np.array([x[0] / x[1] for x in ucom])

        # match peaks to true energies
        cals = {}
        for i, er in enumerate(eratios):

            umatch = np.where( np.isclose(uratios, er, rtol=match_thresh) )
            e1, e2 = ecom[i][0], ecom[i][1]
            if test:
                print(f"\nratio {i} -- e1 {e1:.0f}  e2 {e2:.0f} -- {er:.3f}")

            if len(umatch[0]) == 0:
                continue

            caldists = []
            for ij, j in enumerate(umatch[0]):
                u1, u2 = ucom[j][0], ucom[j][1]
                cal = (e2 - e1) / (u2 - u1)
                cal_maxes = cal * umaxes

                # shift peaks by the amount we would expect if this const were true.
                # compute the distance (in "keV") of the peak that minimizes this.
                dist = 0
                for e_true in epeaks:
                    idx = np.abs(cal_maxes - e_true).argmin()
                    dist += np.abs(cal_maxes[idx] - e_true)
                caldists.append([cal, dist])

                if test:
                    dev = er - uratios[j] # set by match_thresh parameter
                    print(f"{ij}  {u1:-5.0f}  {u2:-5.0f}  {dev:-7.3f}  {cal:-5.2f}")

            # get the cal ratio with the smallest total dist
            caldists = np.array(caldists)
            imin = caldists[:,1].argmin()
            cals[i] = caldists[imin, :]

            if test:
                print(f"best: {imin}  {caldists[imin, 0]:.4f}  {caldists[imin, 1]:.4f}")

        if test:
            print("\nSummary:")
            for ipk in cals:
                e1, e2 = ecom[ipk][0], ecom[ipk][1]
                print(f"{ipk}  {e1:-6.1f}  {e2:-6.1f}  cal {cals[ipk][0]:.5f}")

        # get first-pass const for this DataSet
        cal_vals = np.array([c[1][0] for c in cals.items()])
        ds_cal = np.median(cal_vals)
        ds_std = np.std(cal_vals)
        print(f"Pass-1 cal for {etype}: {ds_cal:.5e} pm {ds_std:.5e}")
    
    # if we get here, we failed
    print('Warning, peakdet has failed.')
    return None, False


def peakdet_input(df_group, config):
    """
    $ ./energy_cal.py -q 'whatever' -pi [input_id] [-p]
    Instead of using the automatic peakdet algorithm, compute the first-guess
    calibration constant from an input file.
    """
    # load data and compute runtime
    dsp_list = config['lh5_dir'] + df_group['dsp_path'] + '/' + df_group['dsp_file']
    edata = lh5.load_nda(dsp_list, config['rawe'], config['input_table'], verbose=False)
    runtime_min = df_group['runtime'].sum()
    run = df_group.run.iloc[0]
    cyclo, cychi = df_group.cycle.iloc[0], df_group.cycle.iloc[-1]
    print(f'  Runtime: {runtime_min:.1f} min.  Calibrating:', [f'{et}:{len(ev)} events' for et, ev in edata.items()])
    
    # loop over energy estimators of interest
    pd_results = {}
    for et in config['rawe']:

        # get histogram, error, normalize by runtime, and derivative
        xlo, xhi, xpb = config['init_vals'][et]['raw_range']
        hist, bins, var = pgh.get_hist(edata[et], range=(xlo, xhi), dx=xpb)
        hist_norm = np.divide(hist, runtime_min * 60)
        hist_err = np.array([np.sqrt(hbin / (runtime_min * 60)) for hbin in hist])

        # load the input peaks
        inp_id = config['input_id'] # string id, like 002
        with open(config['input_peaks']) as f:
            pk_inputs = json.load(f)
        # pprint(pk_inputs)
        pk_list = {k:v for k,v in pk_inputs[inp_id][et].items()}
        yv = [pk_list[k][0] for k in pk_list] # true peaks (keV)
        xv_input = [pk_list[k][1] for k in pk_list] # raw peaks (uncalib.)
        # pprint(pk_list)
        
        # To make the input_peaks method more robust, add a step to refine
        # the input peak guess that can catch small changes in gain.
        # For each peak, select the maximum bin within 3% of the input
        # raw energy value.  It's hard to make this window larger if you're 
        # using calibration peaks very close together (like 583 and 609).
        xv_tuned = []
        for rpk in xv_input:
            winlo, winhi = rpk * (1 - 0.03), rpk * (1 + 0.03)
            idx = np.where((bins >= winlo) & (bins <= winhi))
            ilo = idx[0][0]
            imax = np.argmax(hist_norm[idx])
            ipk = ilo + imax
            xval_adj = bins[ipk]
            xv_tuned.append(xval_adj)
        xv = xv_tuned
        
        # run polyfit (pass-1 fit is simple)
        pol = config['pol'][0]
        pfit = np.polyfit(xv, yv, pol) # p2, p1, p0
        
        pd_results = {}
        pd_results[f'{et}_calpass'] = True
        pd_results[f'{et}_runtime'] = runtime_min
        pd_results[f'{et}_cyclo'] = cyclo
        pd_results[f'{et}_cychi'] = cychi
        for i, p in enumerate(np.flip(pfit)): # p0, p1, p2
            pd_results[f'{et}_pol{i}'] = p

        if config['show_plot']:
            
            # plot uncalibrated and calibrated energy spectrum, w/ maxima
            fig, (p0, p1) = plt.subplots(2, 1, figsize=(8, 8))
            
            # 1. show spectrum and input peaks
            p0.semilogy(bins[1:], hist_norm, 'b', ds='steps', lw=1)
            
            p0.plot(np.nan, np.nan, '-w', label=f'Run {run}, cyc {cyclo}--{cychi}')
            
            cmap = plt.cm.get_cmap('jet', len(pk_list))
            for i in range(len(xv)):
                rpk = xv[i]
                idx = (np.abs(bins - rpk)).argmin()
                p0.plot(rpk, hist_norm[idx], 'v', ms=10, c=cmap(i),
                        label=f'{yv[i]} : {rpk:.0f}')
            
            p0.set_xlabel(f'{et} (uncal)', ha='right', x=1)
            p0.set_ylabel(f'Counts / min / {xpb:.1f} keV', ha='right', y=1)
            p0.legend(fontsize=10)
            
            # 2: show the calibration curve fit result
            p1.plot(np.nan, np.nan, '-w', label=f'Run {run}, cyc {cyclo}--{cychi}')
            
            p1.plot(xv, yv, '.k')
            
            polfunc = np.poly1d(pfit) # handy numpy polynomial function
            yfit = polfunc(xv)
            pol_label = '  '.join([f'p{i} : {ene:.2e}' for i, ene in enumerate(pfit)])
            p1.plot(xv, yfit, '-r', lw=2, label=pol_label)
            
            p1.set_xlabel('Energy (keV)', ha='right', x=1)
            p1.set_ylabel(f'{et} (uncal)', ha='right', y=1)
            p1.legend(fontsize=10)
            
            if config['batch_mode']:
                plt.savefig(f'./plots/energy_cal/peakinput_{et}_run{run}_clo{cyclo}_chi{cychi}.pdf')
            else:
                plt.show()
            plt.close()
            
        return pd.Series(pd_results)


def run_peakfit(dg, config, db_ecal):
    """
    $ ./energy_cal.py -q 'query' -pf [-pi KEY] [-p : show plot]
    
    Using the first guess calibration from peakdet, fit each peak of interest
    and compute the calibration and resolution curves.
    """
    gb = dg.fileDB.groupby(config['gb_cols'])
    run = dg.fileDB.run.iloc[0]
    cyclo, cychi = dg.fileDB.cycle.iloc[0], dg.fileDB.cycle.iloc[-1]
    print(f'Running peakfit, run {run}, cycles {cyclo}--{cychi}')
    
    result = gb.apply(peakfit, *[config, db_ecal])
    
    # write separate tables for each energy estimator to the TinyDB
    for epar in config['rawe']:

        # format output
        epar_cols = [r for r in result.columns if epar in r]
        df_epar = result[epar_cols].copy()
        df_epar.rename(columns={c:c.split('_')[-1] for c in epar_cols},
                       inplace=True)
        df_epar.reset_index(inplace=True)
        df_epar['run'] = df_epar['run'].astype(str)

        if 'cycle' in df_epar.columns:
            # this is redundant with cyclo and cychi
            df_epar.drop('cycle', 1, inplace=True)
        
        tb_name = f'peakfit_{epar}'
        print('Results:', tb_name)
        print(df_epar)
        
        # this is in-memory, no write to file yet
        table = db_ecal.table(tb_name)
        q = db.Query()
        for i, row in df_epar.iterrows():
            que = ((q.run==row.run) & (q.cyclo==row.cyclo) & (q.cychi==row.cychi))
            table.upsert(row.to_dict(), que)

    if not config['write_db']:
        print('Done. ecalDB write mode not set (-w option)')
        return

    # show in-memory state and then write to file
    # pprint(db_ecal.storage.read())
    print('Writing results to ecalDB.')
    pmd.write_pretty(db_ecal.storage.read(), config['ecaldb'])


def peakfit(df_group, config, db_ecal):
    """
    Example:
    $ ./energy_cal.py -q 'run==117' -pf [-pi 002 : use peakinput] [-p : show plot]
    """
    # choose the mode of peakdet to look up constants from 
    if 'input_id' in config.keys():
        pol = config['pol'][0]
        print('  Using 1st-pass constants from peakdet_input')
        input_peaks = True
    else:
        print('  Using 1st-pass constants from peakdet_auto')
        input_peaks = False
        pol = 1 # and p0==0 always
        
    run = df_group.run.iloc[0]
    cyclo, cychi = df_group.cycle.iloc[0], df_group.cycle.iloc[-1]    
    
    gb_run = df_group['run'].unique()
    if len(gb_run) > 1:
        print("Multi-run queries aren't supported yet, sorry!")
        exit()
        
    # load data and compute runtime
    dsp_list = config['lh5_dir'] + df_group['dsp_path'] + '/' + df_group['dsp_file']
    raw_data = lh5.load_nda(dsp_list, config['rawe'], config['input_table'], verbose=False)
    runtime_min = df_group['runtime'].sum()
    print(f'  Runtime: {runtime_min:.1f} min.  Calibrating:', [f'{et}:{len(ev)} events' for et, ev in raw_data.items()])
    print(f'  Fitting to:', config['fit_func'])

    # get list of peaks to look for
    epeaks = config['expected_peaks'] + config['test_peaks']
    epeaks = np.array(sorted(epeaks))
    
    # loop over energy estimators of interest
    pf_results = {}
    for et in config['rawe']:

        # load first-guess calibration constants from tables in the ecalDB
        # convention for p_i : p0  +  p1 * x  +  p2 * x**2  +  ...
        tb_name = f'peakinp_{et}' if input_peaks else f'peakdet_{et}'
        db_table = db_ecal.table(tb_name).all()
        df_cal = pd.DataFrame(db_table)
        que = f'run=={run} and cyclo=={cyclo} and cychi=={cychi}'
        p1cal = df_cal.query(que)
        if len(p1cal) != 1:
            print(f"Can't load a unique set of cal constants!\n  Full cal DF, '{tb_name}':")
            print(df_cal)
            print('Result of query:', que)
            print(p1cal)
            exit()
        cal_pars_init = [p1cal[f'pol{p}'].iloc[0] for p in range(pol, -1, -1)]
        
        # NOTE: polyfit reverses the coefficients, putting highest order first
        cp = [f'p{i} {cp:.4e} ' for i, cp in enumerate(cal_pars_init[::-1])]
        print(f'  First pass inputs:', ' '.join(cp))
        
        # loop over each peak
        fit_results = fit_peaks(epeaks, cal_pars_init, raw_data[et], runtime_min,
                                config['fit_func'], config['verbose'],
                                config['batch_mode'])

        view_cols = ['epk', 'mu', 'fwhm', 'bkg', 'rchisq', 'mu_raw']
        df_fits = pd.DataFrame(fit_results).T
        # print(df_fits)

        # ----------------------------------------------------------------------
        # compute energy calibration by matrix inversion (thanks Tim and Jason!)

        true_peaks = df_fits['epk']
        raw_peaks, raw_error = df_fits['mu_raw'], df_fits['mu_unc']

        error = raw_error / raw_peaks * true_peaks
        cov = np.diag(error**2)
        weights = np.diag(1 / error**2)

        # create the matrix with columns of descending degree
        degree = config['pol'][0]
        raw_peaks_matrix = np.zeros((len(raw_peaks), degree+1))
        for i, pk in enumerate(raw_peaks):
            temp_degree = degree
            row = np.array([])
            while temp_degree >= 0:
                row = np.append(row, pk**temp_degree)
                temp_degree -= 1
            raw_peaks_matrix[i] += row
        # print(raw_peaks_matrix)

        # perform matrix inversion
        xTWX = np.dot(np.dot(raw_peaks_matrix.T, weights), raw_peaks_matrix)
        xTWY = np.dot(np.dot(raw_peaks_matrix.T, weights), true_peaks)
        if np.linalg.det(xTWX) == 0:
            print("singular matrix, determinant is 0, can't get cal constants")
            exit()
        xTWX_inv = np.linalg.inv(xTWX)

        # get polynomial coefficients and error
        cal_pars_best = np.dot(xTWX_inv, xTWY)
        cal_errs = np.sqrt(np.diag(xTWX_inv))
        n = len(cal_pars_best)
        
        cp = [f'p{i} {cp:.4e} ' for i, cp in enumerate(cal_pars_best[::-1])]
        print(f'  Peakfit results:  ', ' '.join(cp))
        
        # ----------------------------------------------------------------------

        # repeat the peak fit with the new 'best' energy (can affect width
        # especially if the peaks are displaced from the guessed locations)
        fit_results = fit_peaks(epeaks, cal_pars_best, raw_data[et], runtime_min,
                                config['fit_func'], config['verbose'],
                                config['batch_mode'])
        
        df_fits = pd.DataFrame(fit_results).T
        
        # compute the difference between lit and measured values
        pfunc = np.poly1d(cal_pars_best)
        cal_data = pfunc(raw_data[et])
        cal_peaks = pfunc(raw_peaks)
        df_fits['residual'] = true_peaks - cal_peaks
        print(df_fits)

        # fit fwhm vs. energy
        # FWHM(E) = sqrt(A_noise^2 + A_fano^2 * E + A_qcol^2 E^2)
        # Ref: Eq. 3 of https://arxiv.org/abs/1902.02299
        # TODO: fix error handling
        def sqrt_fwhm(x, a_n, a_f, a_c):
            return np.sqrt(a_n**2 + a_f**2 * x + a_c**2 * x**2)
        p_guess = [0.3, 0.05, 0.001]
        p_fit, p_cov = curve_fit(sqrt_fwhm, df_fits['mu'], df_fits['fwhm'],
                                 p0=p_guess)#, sigma = np.sqrt(h), absolute_sigma=True)
        p_err = np.sqrt(np.diag(p_cov))

        # show a split figure with calibrated spectrum + used peaks on top,
        # and calib.function and resolution vs. energy on bottom
        if config['show_plot']:
            
            fig = plt.figure(figsize=(8,8))
            p0 = plt.subplot(2, 1, 1) # calibrated spectrum
            p1 = plt.subplot(2, 2, 3) # resolution vs energy
            p2 = plt.subplot(2, 2, 4) # fit_mu vs energy
            
            # 0. show calibrated spectrum with gamma lines
            # get histogram (cts / keV / d)
            xlo, xhi, xpb = config['cal_range']
            hist, bins, _ = pgh.get_hist(cal_data, range=(xlo, xhi), dx=xpb)
            hist_norm = np.divide(hist, runtime_min * 60 * xpb)

            # show peaks
            cmap = plt.cm.get_cmap('brg', len(df_fits)+1)
            for i, row in df_fits.iterrows():

                # get a pretty label for the isotope
                lbl = config['pks'][str(row['epk'])]
                iso = ''.join(r for r in re.findall('[0-9]+', lbl))
                ele = ''.join(r for r in re.findall('[a-z]', lbl, re.I))
                pk_lbl = r'$^{%s}$%s' % (iso, ele)

                pk_diff = row['epk'] - row['mu']
                p0.axvline(row['epk'], ls='--', c=cmap(i), lw=1,
                            label=f"{pk_lbl} : {row['epk']} + {pk_diff:.3f}")

            p0.semilogy(bins[1:], hist_norm, ds='steps', c='b', lw=1)
            p0.set_xlabel('Energy (keV)', ha='right', x=1)
            p0.set_ylabel('cts / s / keV', ha='right', y=1)
            p0.legend(loc=3, fontsize=11)

            # 1. resolution vs. energy
            
            # TODO: add fwhm errorbar
            x_fit = np.arange(xlo, xhi, xpb)
            y_init = sqrt_fwhm(x_fit, *p_guess)
            # p1.plot(x_fit, y_init, '-', lw=1, c='orange', label='guess')

            y_fit = sqrt_fwhm(x_fit, *p_fit)
            a_n, a_f, a_c = p_fit
            fit_label = r'$\sqrt{(%.2f)^2 + (%.3f)^2 E + (%.4f)^2  E^2}$' % (a_n, a_f, a_c)
            p1.plot(x_fit, y_fit, '-r', lw=1, label=f'fit: {fit_label}')

            p1.plot(df_fits['mu'], df_fits['fwhm'], '.b')

            p1.set_xlabel('Energy (keV)', ha='right', x=1)
            p1.set_ylabel('FWHM (keV)', ha='right', y=1)
            p1.legend(fontsize=11)
            
            # 2. fit_mu vs. energy
            p2.plot(df_fits.epk, df_fits.epk - df_fits.mu, '.b', 
                    label=r'$E_{true}$ - $E_{fit}$')
            p2.set_xlabel('Energy (keV)', ha='right', x=1)
            p2.set_ylabel('Residual (keV)', ha='right', y=1)
            p2.legend(fontsize=13)

            if config['batch_mode']:
                print('Saving plot')
                plt.savefig(f'./plots/energy_cal/peakfit_{et}_run{run}_clo{cyclo}_chi{cychi}.pdf')
            else:
                plt.show()
            plt.cla()

        # fill in the peakfit results and return
        
        # cycle range
        pf_results[f'{et}_cyclo'] = cyclo
        pf_results[f'{et}_cychi'] = cychi
        
        # energy calibration constants 
        for i, p in enumerate(cal_pars_best[::-1]): # remember to flip the coeffs!
            pf_results[f'{et}_cal{i}'] = p

        # uncertainties in cal constants
        for i, pe in enumerate(cal_errs[::-1]):
            pf_results[f'{et}_unc{i}'] = pe
        
        # resolution curve parameters
        pf_results[f'{et}_Anoise'] = p_fit[0]
        pf_results[f'{et}_Afano'] = p_fit[1]
        pf_results[f'{et}_Aqcol'] = p_fit[2]
        pf_results[f'{et}_runtime'] = runtime_min
        
    return pd.Series(pf_results)


def fit_peaks(epeaks, cal_pars, raw_data, runtime_min, ff_name='gauss_step', show_plot=True, batch_mode=False):
    """
    Routine for sequential fit of peaks.
    
    Inputs: 
    - epeaks: list of peak energies to calibrate, e.g. [1460, 2615, ...]
    - cal_pars: results from peakdet for the first estimate of the calibration:
        cal_data = p0  +  p1 * raw_data  +  p2 * raw_data**2  +  ...
    - raw_data: numpy array of uncalibrated data.  The array is needed instead 
        of a histogram because this routine tries to optimize the binning
        around each peak.
    - runtime_min : this is used to normalize spectra to cts/min, which helps a
        lot to compute initial guesses for fit functions.
    
    Returns a dict, 'fit_results', which is easily convertible to DataFrame.
    """
    # compute calibrated energy (1st time)
    pfunc = np.poly1d(cal_pars) # handy numpy polynomal function
    cal_data = pfunc(raw_data)
    
    # quick spectrum check (check that sure peakdet's guess is in the ballpark)
    if show_plot:
        xlo, xhi, xpb = 0, 3000, 5
        hist, bins, _ = pgh.get_hist(cal_data, range=(xlo, xhi), dx=xpb)
        hist_norm = np.divide(hist, runtime_min * 60 * xpb)
        plt.semilogy(bins[1:], hist_norm, ds='steps', c='b', lw=1)
        plt.show()
        plt.cla()
    
    # loop over peak energies
    fit_results = {}
    for ie, epk in enumerate(epeaks):
        
        # adjust the window.  resolution goes as roughly sqrt(energy)
        window = np.sqrt(epk)
        xlo, xhi = epk - window / 2, epk + window / 2
        nbins = int(window) * 3 # todo, make this get smaller w/ inc energy
        xpb = (xhi - xlo) / nbins
        
        if show_plot:
            print(f'Fitting peak at {epk:6} keV.  xlo {xlo:6.1f}  xhi {xhi:6.1f}  xpb {xpb:.3f}  nbins {nbins}')
        
        # get histogram, error, normalize by runtime
        pk_data = cal_data[(cal_data >= xlo) & (cal_data <= xhi)]
        hist, bins, _ = pgh.get_hist(pk_data, range=(xlo, xhi), dx=xpb)
        hist_norm = np.divide(hist, runtime_min * 60)
        hist_var = np.array([np.sqrt(h / (runtime_min * 60)) for h in hist])
        
        # estimate left and right sideband locations
        ibkg_lo, ibkg_hi = int(nbins * 0.2), int(nbins * 0.8)
        bkg0 = np.mean(hist_norm[ :ibkg_lo])
        bkg0_hi = np.mean(hist_norm[ibkg_hi:])
        b, h = bins[1:], hist_norm - bkg0
        
        # default: gaussian fit + step function : a, mu, sigma, bkg, step
        if ff_name == 'gauss_step':
            
            fit_func = pgf.gauss_step    

            # set robust initial guesses
            step0 = bkg0 - bkg0_hi
            imax = np.argmax(h)
            upr_half = b[np.where((b > b[imax]) & (h <= np.amax(h)/2))][0]
            bot_half = b[np.where((b < b[imax]) & (h <= np.amax(h)/2))][-1]
            fwhm0 = upr_half - bot_half
            sig0 = fwhm0 / 2.355
            amp0 = np.amax(h) * fwhm0
            p_init = [amp0, bins[imax], sig0, bkg0, step0]

            p_fit, p_cov = pgf.fit_hist(fit_func, hist_norm, bins, 
                                        var=hist_var, guess=p_init)
            p_err = np.sqrt(np.diag(p_cov))
            fwhm = p_fit[2] * 2.355
            fwhm_err = p_err[2] * 2.355 * epk / p_fit[2]

            fit_results[ie] = {
                'epk':epk, 'mu':p_fit[1], 'fwhm':p_fit[2]*2.355,
                'sig':p_fit[2], 'amp':p_fit[0], 'bkg':p_fit[3]
                }

        # peakshape : mu, sigma, hstep, htail, tau, bg0, amp
        # this requires higher stats, doesn't work as well for smaller peaks
        elif ff_name == 'peakshape':
            
            fit_func = pgf.radford_peak
            
            # set robust initial guesses
            step0 = bkg0 - bkg0_hi
            imax = np.argmax(h)
            upr_half = b[np.where((b > b[imax]) & (h <= np.amax(h)/2))][0]
            bot_half = b[np.where((b < b[imax]) & (h <= np.amax(h)/2))][-1]
            fwhm0 = upr_half - bot_half
            sig0 = fwhm0 / 2.355
            amp0 = np.amax(h) * fwhm0
            htail, tau = 0.1, 10 # TODO: find a way to guess these
            p_init = [bins[imax], sig0, step0, htail, tau, bkg0, amp0]
            
            p_fit, p_cov = pgf.fit_hist(fit_func, hist_norm, bins,
                                        var=hist_var, guess=p_init)
            p_err = np.sqrt(np.diag(p_cov))
            fwhm = p_fit[1] * 2.355
            fwhm_err = p_err[1] * 2.355 * epk / p_fit[1]
            
            fit_results[ie] = {
                'epk':epk,
                'mu':p_fit[0], 'fwhm':p_fit[1]*2.355, 'sig':p_fit[1],
                'amp':p_fit[6], 'bkg':p_fit[5]
                }
            
        # compute goodness of fit
        rchisq = pgf.goodness_of_fit(hist_norm, b, fit_func, p_fit)
        fit_results[ie]['rchisq'] = rchisq
            
        # Now we need to invert the cal polynomial to get the raw position
        # of the peak that we've found.  This allows us to refine
        # an initial estimate before computing a final resolution curve, 
        # just by calling fit_peaks multiple times.
        # But the polynomial has multiple roots!  How to make this automatic?
        # Trick: pick the root that most closely matches what you would get
        # by only consdering our 1st-order calibration term from peakdet.
        # For a fairly linear system like a Ge detector this should work well.
        pk_guess = fit_results[ie]['mu'] / cal_pars[1]
        pk_roots = (pfunc - epk).roots
        ipk_closest = (np.abs(pk_roots - pk_guess)).argmin()
        mu_raw = pk_roots[ipk_closest]
        mu_unc = p_err[1] * (mu_raw / epk)
        fit_results[ie]['mu_raw'], fit_results[ie]['mu_unc'] = mu_raw, mu_unc
        
        # print(epk, (pfunc - epk).roots, p_fit[1] / cal_pars[1] )
        # print(ipk_closest)
        # print(mu_raw, mu_unc)
        
        if show_plot:
            xfit = np.arange(xlo, xhi, xpb * 0.1)
            plt.axvline(bins[ibkg_lo], c='m', label='bkg region')
            plt.plot(xfit, fit_func(xfit, *p_init), '-', c='orange', label='init')
            plt.plot(xfit, fit_func(xfit, *p_fit), '-', c='red', label='fit')
            plt.plot(bins[1:], hist_norm, c='b', lw=1.5, ds='steps')
            plt.plot(np.nan, np.nan, 'w', label=f'FWHM: {fwhm:.2f}')
            plt.xlabel('pass-1 energy (kev)', ha='right', x=1)
            plt.legend(fontsize=12)
            if batch_mode:
                plt.savefig(f'./plots/energy_cal/fit{ie}_peakfit.png')
            else:
                plt.show()
            plt.close()
            
    return fit_results


if __name__=='__main__':
    main()
