#!/usr/bin/env python3
import os
import re
import time
import json
import argparse
import pandas as pd
import numpy as np
from pprint import pprint
from datetime import datetime
import itertools
from collections import OrderedDict
from scipy.optimize import curve_fit

import tinydb as db
from tinydb.storages import MemoryStorage

import matplotlib
if os.environ.get('HOSTNAME'): # cenpa-rocks
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('../clint.mpl')
from matplotlib.colors import LogNorm

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from tqdm import tqdm
    tqdm.pandas() # suppress annoying FutureWarning

from pygama import DataGroup
from pygama.dsp.units import *
from pygama.io.raw_to_dsp import build_processing_chain
from pygama.dsp.ProcessingChain import ProcessingChain
import pygama.io.lh5 as lh5
import pygama.analysis.histograms as pgh
import pygama.analysis.peak_fitting as pgf


def main():
    doc="""
    === optimizer.py ====================================================

    dsp optimization app, works with DataGroup

    === C. Wiseman (UW) =============================================
    """
    rthf = argparse.RawTextHelpFormatter
    par = argparse.ArgumentParser(description=doc, formatter_class=rthf)
    arg, st, sf = par.add_argument, 'store_true', 'store_false'
    
    # primary operations
    arg('-q', '--query', nargs=1, type=str,
        help="select group to analyze: -q 'cycle==1' ")
    arg('-e', '--energy', action=st, help='optimize energy trapezoid')
    arg('-d', '--dcr', action=st, help='optimize DCR parameter')

    args = par.parse_args()
    
    # -- setup -- 
    
    # load main DataGroup, select files to analyze
    dg = DataGroup('cage.json', load=True)
    if args.query:
        que = args.query[0]
        dg.fileDB.query(que, inplace=True)
    else:
        dg.fileDB = dg.fileDB[-1:]
 
    view_cols = ['run','cycle','daq_file','runtype','startTime','threshold']
    print(dg.fileDB[view_cols].to_string())
    # print(f'Found {len(dg.fileDB)} files.')
    
    # -- run routines -- 
    
    # TODO : we could split this code into "spectrum" (peak width) optimizations, 
    # and "waveform" optimizations, where the FOM is a waveform, not a peak.
    # so like optimize_spec.py and optimize_wfs.py
    
    optimize_trap(dg)
    show_trap_results()
    
    # optimize_dcr(dg) 
    # show_dcr_results(dg)
    # check_wfs(dg)
    
    
def optimize_trap(dg):
    """
    Generate a file with grid points to search, and events from the target peak.  
    Then run DSP a bunch of times on the small table, and fit the peak w/ the
    peakshape function.  
    NOTE: run table-to-table DSP (no file I/O)
    """
    f_peak = './temp_peak.lh5' # lh5
    f_results = './temp_results.h5' # pandas
    grp_data, grp_grid = '/optimize_data', '/optimize_grid'
    
    # epar, elo, ehi, epb = 'energy', 0, 1e7, 10000 # full range
    epar, elo, ehi, epb = 'energy', 3.88e6, 3.92e6, 500 # K40 peak
    
    show_movie = True
    write_output = True
    n_rows = None # default None
    
    with open('opt_trap.json') as f:
        dsp_config = json.load(f, object_pairs_hook=OrderedDict)
    
    # files to consider.  fixme: right now only works with one file
    sto = lh5.Store()
    lh5_dir = os.path.expandvars(dg.config['lh5_dir'])
    raw_list = lh5_dir + dg.fileDB['raw_path'] + '/' + dg.fileDB['raw_file']
    f_raw = raw_list.values[0] 
    tb_raw = 'ORSIS3302DecoderForEnergy/raw/'

    # quick check of the energy range
    # ene_raw = sto.read_object(tb_raw+'/'+epar, f_raw).nda
    # hist, bins, var = pgh.get_hist(ene_raw, range=(elo, ehi), dx=epb)
    # plt.plot(bins[1:], hist, ds='steps')
    # plt.show()
    # exit()
    
    # set grid parameters
    # TODO: jason's suggestions, knowing the expected shape of the noise curve
    # e_rises = np.linspace(-1, 0, sqrt(sqrt(3))
    # e_rises # make another list which is 10^pwr of this list
    # np.linspace(log_tau_min, log_tau_max) # try this too
    e_rises = np.arange(1, 12, 1)
    e_flats = np.arange(1, 6, 1)
    # rc_consts = np.arange(54, 154, 10) # changing this here messes up DCR
    
    # -- create the grid search file the first time -- 
    # NOTE: this makes a linear grid, and is editable by the arrays above.
    # jason also proposed a more active gradient-descent style search
    # like with Brent's method. (https://en.wikipedia.org/wiki/Brent%27s_method)
    
    if True:
    # if not os.path.exists(f_peak):
        print('Recreating grid search file')
        
        # create the grid file
        # NOTE: save it as an lh5 Table just as an example of writing/reading one
        lists = [e_rises, e_flats]#, rc_consts]
        prod = list(itertools.product(*lists)) # clint <3 stackoverflow
        df_grid = pd.DataFrame(prod, columns=['rise', 'flat'])#,'rc']) 
        lh5_grid = {}
        for i, dfcol in df_grid.iteritems():
            lh5_grid[dfcol.name] = lh5.Array(dfcol.values)
        tb_grid = lh5.Table(col_dict=lh5_grid)
        sto.write_object(tb_grid, grp_grid, f_peak)
            
        # filter events by onboard energy
        ene_raw = sto.read_object(tb_raw+'/'+epar, f_raw).nda
        # hist, bins, var = pgh.get_hist(ene_raw, range=(elo, ehi), dx=epb)
        # plt.plot(bins[1:], hist, ds='steps')
        # plt.show()
        if n_rows is not None:
            ene_raw = ene_raw[:n_rows]
        idx = np.where((ene_raw > elo) & (ene_raw < ehi))

        # create a filtered table with correct waveform and attrs
        # TODO: move this into a function in lh5.py which takes idx as an input
        tb_data, wf_tb_data = lh5.Table(), lh5.Table()

        # read non-wf cols (lh5 Arrays)
        data_raw = sto.read_object(tb_raw, f_raw, n_rows=n_rows)
        for col in data_raw.keys():
            if col=='waveform': continue
            newcol = lh5.Array(data_raw[col].nda[idx], attrs=data_raw[col].attrs)
            tb_data.add_field(col, newcol)
        
        # handle waveform column (lh5 Table)
        data_wfs = sto.read_object(tb_raw+'/waveform', f_raw, n_rows=n_rows)
        for col in data_wfs.keys():
            attrs = data_wfs[col].attrs
            if isinstance(data_wfs[col], lh5.ArrayOfEqualSizedArrays):
                # idk why i can't put the filtered array into the constructor
                aoesa = lh5.ArrayOfEqualSizedArrays(attrs=attrs, dims=[1,1])
                aoesa.nda = data_wfs[col].nda[idx]
                newcol = aoesa
            else:
                newcol = lh5.Array(data_wfs[col].nda[idx], attrs=attrs)
            wf_tb_data.add_field(col, newcol)
        tb_data.add_field('waveform', wf_tb_data)
        tb_data.attrs = data_raw.attrs
        sto.write_object(tb_data, grp_data, f_peak)

    else:
        print('Loading peak file. groups:', sto.ls(f_peak))
        tb_grid = sto.read_object(grp_grid, f_peak)
        tb_data = sto.read_object(grp_data, f_peak) # filtered file
        # tb_data = sto.read_object(tb_raw, f_raw) # orig file
        df_grid = tb_grid.get_dataframe()
        
    # check shape of input table
    print('input table attributes:')
    for key in tb_data.keys():
        obj = tb_data[key]
        if isinstance(obj, lh5.Table):
            for key2 in obj.keys():
                obj2 = obj[key2]
                print('  ', key, key2, obj2.nda.shape, obj2.attrs)
        else:
            print('  ', key, obj.nda.shape, obj.attrs)

    # clear new colums if they exist
    new_cols = ['e_fit', 'fwhm_fit', 'rchisq', 'xF_err', 'fwhm_ovr_mean']
    for col in new_cols:
        if col in df_grid.columns:
            df_grid.drop(col, axis=1, inplace=True)

    t_start = time.time()
    def run_dsp(dfrow):
        """
        run dsp on the test file, editing the processor list
        alternate idea: generate a long list of processors with different names
        """
        # adjust dsp config dictionary
        rise, flat = dfrow
        # dsp_config['processors']['wf_pz']['defaults']['db.pz.tau'] = f'{tau}*us'
        dsp_config['processors']['wf_trap']['args'][1] = f'{rise}*us'
        dsp_config['processors']['wf_trap']['args'][2] = f'{flat}*us'
        # pprint(dsp_config)
        
        # run dsp
        pc, tb_out = build_processing_chain(tb_data, dsp_config, verbosity=0)
        pc.execute()
        
        # analyze peak
        e_peak = 1460.
        etype = 'trapEmax'
        elo, ehi, epb = 4000, 4500, 3 # the peak moves around a bunch
        energy = tb_out[etype].nda
        
        # get histogram
        hE, bins, vE = pgh.get_hist(energy, range=(elo, ehi), dx=epb)
        xE = bins[1:]
        
        # should I center the max at 1460?

        # simple numerical width
        i_max = np.argmax(hE)
        h_max = hE[i_max]
        upr_half = xE[(xE > xE[i_max]) & (hE <= h_max/2)][0]
        bot_half = xE[(xE < xE[i_max]) & (hE >= h_max/2)][0]
        fwhm = upr_half - bot_half
        sig = fwhm / 2.355
        
        # fit to gaussian: amp, mu, sig, bkg
        fit_func = pgf.gauss_bkg
        amp = h_max * fwhm
        bg0 = np.mean(hE[:20])
        x0 = [amp, xE[i_max], sig, bg0]
        xF, xF_cov = pgf.fit_hist(fit_func, hE, bins, var=vE, guess=x0)

        # collect results
        e_fit = xF[0]
        xF_err = np.sqrt(np.diag(xF_cov))
        e_err = xF
        fwhm_fit = xF[1] * 2.355 * 1460. / e_fit
        
        fwhm_err = xF_err[2] * 2.355 * 1460. / e_fit
        
        chisq = []
        for i, h in enumerate(hE):
            model = fit_func(xE[i], *xF)
            diff = (model - h)**2 / model
            chisq.append(abs(diff))
        rchisq = sum(np.array(chisq) / len(hE))
        fwhm_ovr_mean = fwhm_fit / e_fit

        if show_movie:
            
            plt.plot(xE, hE, ds='steps', c='b', lw=2, label=f'{etype} {rise}--{flat}')

            # peak shape
            plt.plot(xE, fit_func(xE, *x0), '-', c='orange', alpha=0.5,
                     label='init. guess')
            plt.plot(xE, fit_func(xE, *xF), '-r', alpha=0.8, label='peakshape fit')
            plt.plot(np.nan, np.nan, '-w', label=f'mu={e_fit:.1f}, fwhm={fwhm_fit:.2f}')

            plt.xlabel(etype, ha='right', x=1)
            plt.ylabel('Counts', ha='right', y=1)
            plt.legend(loc=2)

            # show a little movie
            plt.show(block=False)
            plt.pause(0.01)
            plt.cla()

        # return results
        return pd.Series({'e_fit':e_fit, 'fwhm_fit':fwhm_fit, 'rchisq':rchisq,
                          'fwhm_err':xF_err[0], 'fwhm_ovr_mean': fwhm_ovr_mean})
    
    # df_grid=df_grid[:10]
    df_tmp = df_grid.progress_apply(run_dsp, axis=1)
    df_grid[new_cols] = df_tmp
    # print(df_grid)
    
    if show_movie:
        plt.close()
    
    print('elapsed:', time.time() - t_start)
    if write_output:
        df_grid.to_hdf(f_results, key=grp_grid)
        print(f"Wrote output file: {f_results}")


def show_trap_results():
    """
    plot of ramp/flat time vs target peak FWHM
    """
    df_grid = pd.read_hdf('./temp_results.h5', '/optimize_grid')
    print(df_grid)
    
    print('Minimum fwhm:')
    print(df_grid[df_grid.fwhm_ovr_mean==df_grid.fwhm_ovr_mean.min()])
    
    plt.plot(df_grid.e_fit, df_grid.fwhm_ovr_mean, '.b')
    plt.show()
    
    
def optimize_dcr(dg):
    """
    I don't have an a priori figure of merit for the DCR parameter, until I can
    verify that we're seeing alphas.  So this function should just run processing
    on a CAGE run with known alpha events, and show you the 2d DCR vs. energy.
    
    Once we know we can reliably measure the alpha distribution somehow, then
    perhaps we can try a grid search optimization like the one done in 
    optimize_trap.
    """
    # files to consider.  fixme: right now only works with one file
    sto = lh5.Store()
    lh5_dir = os.path.expandvars(dg.config['lh5_dir'])
    raw_list = lh5_dir + dg.fileDB['raw_path'] + '/' + dg.fileDB['raw_file']
    f_raw = raw_list.values[0] 
    
    tb_raw = 'ORSIS3302DecoderForEnergy/raw/'
    tb_data = sto.read_object(tb_raw, f_raw)
    
    cycle = dg.fileDB['cycle'].values[0]
    f_results = f'./temp_{cycle}.h5'
    
    write_output = True
    
    # adjust dsp config 
    with open('opt_dcr.json') as f:
        dsp_config = json.load(f, object_pairs_hook=OrderedDict)
    # pprint(dsp_config)
    # exit()
    
    # set dcr parameters
    # rise, flat, dcr_tstart = 200, 1000, 'tp_0+1.5*us' # default
    # dcr_rise, dcr_flat, dcr_tstart = 100, 3000, 'tp_0+3*us' # best so far?
    dcr_rise, dcr_flat, dcr_tstart = 100, 2500, 'tp_0+1*us'
    dsp_config['processors']['dcr_raw']['args'][1] = dcr_rise
    dsp_config['processors']['dcr_raw']['args'][2] = dcr_flat
    dsp_config['processors']['dcr_raw']['args'][3] = dcr_tstart
    
    # set trap energy parameters
    # ene_rise, ene_flat = "2*us", "1*us" # best? from optimize_trap
    ene_rise, ene_flat = "10*us", "5*us"
    dsp_config['processors']['wf_trap']['args'][1] = ene_rise
    dsp_config['processors']['wf_trap']['args'][2] = ene_flat
    
    # adjust pole-zero constant
    dsp_config['processors']['wf_pz']['defaults']['db.pz.tau'] = '64.4*us'
    # dsp_config['processors']['wf_pz']['defaults']['db.pz.tau'] = '50*us'
    # dsp_config['processors']['wf_pz']['defaults']['db.pz.tau'] = '100*us'
    
    # run dsp
    print('Running DSP ...')
    t_start = time.time()
    pc, tb_out = build_processing_chain(tb_data, dsp_config, verbosity=1)
    pc.execute()
    t_elap = (time.time() - t_start)/60
    print(f'Done.  Elapsed: {t_elap:.2f} min')
        
    df_out = tb_out.get_dataframe()
    
    if write_output:
        df_out.to_hdf(f_results, key='opt_dcr')
        print('Wrote output file:', f_results)
    
    
def show_dcr_results(dg):
    """
    plot of dcr vs energy for a single setting
    """
    cycle = dg.fileDB['cycle'].values[0]
    df_dsp = pd.read_hdf(f'./temp_{cycle}.h5', 'opt_dcr')
    # print(df_dsp.describe())    

    # compare DCR and A/E distributions
    fig, (p0, p1) = plt.subplots(2, 1, figsize=(8, 8))
    
    elo, ehi, epb = 0, 25000, 100
    
    # aoe distribution
    # ylo, yhi, ypb = -1, 2, 0.1
    # ylo, yhi, ypb = -0.1, 0.3, 0.005
    ylo, yhi, ypb = 0.05, 0.08, 0.0005
    nbx = int((ehi-elo)/epb)
    nby = int((yhi-ylo)/ypb)
    h = p0.hist2d(df_dsp['trapEmax'], df_dsp['aoe'], bins=[nbx,nby],
                   range=[[elo, ehi], [ylo, yhi]], cmap='jet',
                   norm=LogNorm())
    # p0.set_xlabel('Energy (uncal)', ha='right', x=1)
    p0.set_ylabel('A/E', ha='right', y=1)

    # dcr distribution
    # ylo, yhi, ypb = -20, 20, 1 # dcr_raw
    # ylo, yhi, ypb = -5, 2.5, 0.1 # dcr = dcr_raw / trapEmax
    # ylo, yhi, ypb = -3, 2, 0.1
    ylo, yhi, ypb = 0.9, 1.08, 0.001
    ylo, yhi, ypb = 1.034, 1.0425, 0.00005 # best for 64.4 us pz
    # ylo, yhi, ypb = 1.05, 1.056, 0.00005 # best for 50 us pz
    # ylo, yhi, ypb = 1.016, 1.022, 0.00005 # best for 100 us pz
    nbx = int((ehi-elo)/epb)
    nby = int((yhi-ylo)/ypb)
    h = p1.hist2d(df_dsp['trapEmax'], df_dsp['dcr'], bins=[nbx,nby],
                   range=[[elo, ehi], [ylo, yhi]], cmap='jet',
                   norm=LogNorm())
    p1.set_xlabel('Energy (uncal)', ha='right', x=1)
    p1.set_ylabel('DCR', ha='right', y=1)
    
    # plt.show()
    plt.savefig(f'./plots/dcr_cyc{cycle}.png', dpi=300)
    plt.cla()
    
    
def check_wfs(dg):
    """
    somebody inevitably asks you, 'have you looked at the waveforms?'
    in this function, compare alpha wfs to gamma wfs
    
    use the temp_results file to pick indexes, and grab the corresponding
    wfs.  LH5 doesn't let us only load particular indexes (yet), so we
    have to load all the waveforms in the file every time.  butts.
    """
    # load dsp results
    cycle = dg.fileDB['cycle'].values[0]
    df_dsp = pd.read_hdf(f'./temp_{cycle}.h5', 'opt_dcr')
    
    # load waveforms
    sto = lh5.Store()
    lh5_dir = os.path.expandvars(dg.config['lh5_dir'])
    raw_list = lh5_dir + dg.fileDB['raw_path'] + '/' + dg.fileDB['raw_file']
    f_raw = raw_list.values[0] 
    tb_wfs = sto.read_object('ORSIS3302DecoderForEnergy/raw/waveform', f_raw)
    
    # energy cut
    et = 'trapEmax'
    # elo, ehi = 8000, 16000
    # elo, ehi = 8000, 10000
    elo, ehi = 12000, 13000
    
    # dcr cut
    # alp_lo, alp_hi = -0.5, 0.5
    # gam_lo, gam_hi = 0.8, 1.2
    
    # aoe cut
    alp_lo, alp_hi = 0.064, 0.068
    gam_lo, gam_hi = 0.05, 0.06
    
    # selection
    idx_alp = df_dsp[et].loc[(df_dsp[et] > elo) & (df_dsp[et] < ehi) & 
                             (df_dsp.aoe > alp_lo) & (df_dsp.aoe < alp_hi)].index
    
    idx_gam = df_dsp[et].loc[(df_dsp[et] > elo) & (df_dsp[et] < ehi) & 
                             (df_dsp.aoe > gam_lo) & (df_dsp.aoe < gam_hi)].index
    
    wfs_alp = tb_wfs['values'].nda[idx_alp]
    wfs_gam = tb_wfs['values'].nda[idx_gam]
    
    print(f'found {wfs_alp.shape[0]} alpha candidates')
    print(f'found {wfs_gam.shape[0]} gamma candidates')
    
    # plot 
    # fig, (p0, p1) = plt.subplots(2, 1, figsize=(8, 8))
    
    ts = np.arange(0, wfs_gam.shape[1], 1)

    n_gam = 10 if wfs_gam.shape[0] > 10 else wfs_gam.shape[0]
    for iwf in range(n_gam):
        max = np.amax(wfs_gam[iwf,:])
        # max = df_dsp[et].values[iwf]
        plt.plot(ts[:-1], wfs_gam[iwf,:-1]/max, '-b', lw=1, alpha=0.5)
    
    n_alp = 10 if wfs_alp.shape[0] > 10 else wfs_alp.shape[0]
    for iwf in range(n_alp):
        max = np.amax(wfs_alp[iwf,:])
        # max = df_dsp[et].values[iwf]
        plt.plot(ts[:-1], wfs_alp[iwf,:-1]/max, '-r', lw=1, alpha=0.5)

    # plt.xlim(1

    plt.xlabel('time (clock ticks)', ha='right', x=1)
    plt.ylabel('ADC', ha='right', y=1)
    plt.show()


if __name__=="__main__":
    main()