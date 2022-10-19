#!/usr/bin/env python3
import os
import json
import h5py
import argparse
import pandas as pd
import numpy as np
import tinydb as db
from tinydb.storages import MemoryStorage
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm

import scipy.stats as stats
import scipy.signal as signal

import pygama
from pygama import DataGroup
import pygama.dsp.processors as dsp
import pygama.lh5 as lh5
import pygama.analysis.histograms as pgh
import pygama.analysis.peak_fitting as pgf

def getDataFrame(run, user=True, hit=True, cal=True, lowE=False, dsp_list=[]):
    """
    use DataGroup to get the data of interest and return a pandas dataframe, with other information relevant to the
    data run
    """

    # get run files
    dg = DataGroup('$CAGE_SW/processing/cage.json', load=True)
    str_query = f'run=={run} and skip==False'
    dg.fileDB.query(str_query, inplace=True)

    #get runtime, startime, runtype
    runtype_list = np.array(dg.fileDB['runtype'])
    runtype = runtype_list[0]
    rt_min = dg.fileDB['runtime'].sum()
    u_start = dg.fileDB.iloc[0]['startTime']
    t_start = pd.to_datetime(u_start, unit='s')


    # get scan position

    if runtype == 'alp':
        alphaDB = pd.read_hdf(os.path.expandvars('$CAGE_SW/processing/alphaDB.h5'))
        scan_pos = alphaDB.loc[alphaDB['run']==run]
        radius = np.array(scan_pos['radius'])[0]
        angle = np.array(scan_pos['source'])[0]
        rotary = np.array(scan_pos['rotary'])[0]
        #radius = int(radius)
        angle_det = int((-1*angle) - 90)
        if rotary <0:
            angle_det = int(angle + 270)
        print(f'Radius: {radius}; Angle: {angle_det}; Rotary: {rotary}')

    else:
        radius = 'n/a'
        angle = 'n/a'
        angle_det = 'n/a'
        rotary = 'n/a'


    # print(etype, etype_cal, run)
    # exit()

    print(f'user: {user}; cal: {cal}; hit: {hit}')



    # get data and load into df
    lh5_dir = dg.lh5_user_dir if user else dg.lh5_dir

    if cal==True:
        default_dsp_list = ['energy', 'trapEmax', 'trapEftp', 'trapEftp_cal', 'bl','bl_sig', 'bl_slope', 'lf_max', 'A_10','AoE', 'dcr', 'tp_0', 'tp_10', 'tp_90', 'tp_50', 'tp_80', 'tp_max', 'ToE']

    else:
        default_dsp_list = ['energy', 'trapEmax', 'trapEftp', 'bl','bl_sig', 'bl_slope', 'lf_max', 'A_10','AoE', 'dcr', 'tp_0', 'tp_10', 'tp_90', 'tp_50', 'tp_80', 'tp_max', 'ToE']

    if len(dsp_list) < 1:
        print(f'No options specified for DSP list! Using default: {default_dsp_list}')
        dsp_list = default_dsp_list



    if hit==True:
        print('Using hit files')
        file_list = lh5_dir + dg.fileDB['hit_path'] + '/' + dg.fileDB['hit_file']

        if lowE==True:
            file_list = lh5_dir + dg.fileDB['hit_path'] + '/lowE/' + dg.fileDB['hit_file']
            print(f'Using lowE calibration files \n {file_list}')

        if cal==True:
            df = lh5.load_dfs(file_list, dsp_list, 'ORSIS3302DecoderForEnergy/hit')


        if cal==False:
            df = lh5.load_dfs(file_list, dsp_list, 'ORSIS3302DecoderForEnergy/hit')


    elif hit==False:
        print('Using dsp files')
        file_list = lh5_dir + dg.fileDB['dsp_path'] + '/' + dg.fileDB['dsp_file']
        if cal==True:
            df = lh5.load_dfs(file_list, dsp_list, 'ORSIS3302DecoderForEnergy/dsp')
        if cal==False:
            df = lh5.load_dfs(file_list, dsp_list, 'ORSIS3302DecoderForEnergy/dsp')

    else:
        print('dont know what to do here! need to specify if working with calibrated/uncalibrated data, or dsp/hit files')



    return(df, dg, runtype, rt_min, radius, angle_det, rotary)

def apply_DC_Cuts(run, df, cut_keys=set()):

    default_cut_keys = set(['wf_max_cut', 'bl_mean_cut_raw', 'bl_slope_cut_raw', 'bl_sig_cut_raw', 'ftp_max_cut_raw'])



    if len(cut_keys) < 1:
        print(f'No options specified for cut selection! Using default: {default_cut_keys}')
        cut_keys = default_cut_keys

    if 'ftp_max_cut_raw' in cut_keys:
        df['ftp_max'] = df['trapEftp']/df['trapEmax']

    with open(os.path.expandvars('$CAGE_SW/analysis/cuts.json')) as f:
        cuts = json.load(f)

    # always apply muon cut
    df = df.query(cuts[str(run)]['muon_cut']).copy()
    df_cut = df

    total_counts = len(df)
    print(f'total counts before cuts: {total_counts}')


    #have to apply cuts individually instead of using `join` for the set because the total cut string is too long for the query :'(
    for cut in cut_keys:
        print(f'applying cut: {cut}')
        df_cut = df_cut.query((cuts[str(run)][cut])).copy()
        cut_counts = len(df.query((cuts[str(run)][cut])).copy())
        percent_surviving = (cut_counts/total_counts)*100.
        print(f'Percentage surviving {cut} cut: {percent_surviving:.2f}')

    cut_counts_total = len(df_cut)
    percent_surviving_total = (cut_counts_total/total_counts)*100.
    print(f'Percentage surviving cuts: {percent_surviving_total:.2f}')

    return(df_cut)

def getStartStop(run):
    """
    get the start time and stop time for a given run
    """

    # get run files
    dg = DataGroup('$CAGE_SW/processing/cage.json', load=True)
    str_query = f'run=={run} and skip==False'
    dg.fileDB.query(str_query, inplace=True)

    #get runtime, startime, runtype
    runtype_list = np.array(dg.fileDB['runtype'])
    runtype = runtype_list[0]
    rt_min = dg.fileDB['runtime'].sum()
    u_start = dg.fileDB.iloc[0]['startTime']
    t_start = pd.to_datetime(u_start, unit='s')

    u_stop = u_start + rt_min*60

    t_stop = pd.to_datetime(u_stop, unit='s')

    print(f'start: {t_start}\n stop: {t_stop}')
    return(t_start, t_stop)

def corrDCR(df, etype, e_bins=300, elo=0, ehi=6000, dcr_fit_lo=-30, dcr_fit_hi=30):
    """
    Linear fit to DCR to correct for slope. Works for early CAGE runs
    """

    df_dcr_cut = df.query(f'dcr >{dcr_fit_lo} and dcr < {dcr_fit_hi} and {etype} > {elo} and {etype} < {ehi}').copy()


    median, xedges, binnumber = stats.binned_statistic(df_dcr_cut[etype], df_dcr_cut['dcr'], statistic = "median", bins = e_bins)

    en_bin_centers = pgh.get_bin_centers(xedges)

    if quad:
        qmedian, qxedges, qbinnumber = stats.binned_statistic(df_cut[etype], df_cut['dcr'], statistic = "median", bins = int((dcr_fit_qhi-dcr_fit_qlo)/5), range=[dcr_fit_qlo, dcr_fit_qhi])
        qen_bin_centers = pgh.get_bin_centers(qxedges)
        xen = np.concatenate((en_bin_centers, qen_bin_centers))
        ymed = np.concatenate((median, qmedian))
        print(xen)
        print(ymed)
        plt.plot(xen, ymed)
        fit_raw = np.polyfit(xen, ymed, deg=2)
        qconst = fit_raw[0]
        qlin = fit_raw[1]
        qoffset = fit_raw[2]
        df_cut['dcr_corr'] = df_cut['dcr'] - (qconst*(df_cut[etype]**2) + qlin*df_cut[etype] + qoffset)
    else:
        fit_raw = np.polyfit(en_bin_centers, median, deg=1)
        const = fit_raw[0]
        offset = fit_raw[1]

        df_cut['dcr_corr'] = df_cut['dcr'] - (const*(df_cut[etype]) + offset)

    return df_cut

def mode_hist(df, param, a_bins=1000, alo=0.005, ahi=0.075, cut=False, cut_str=''):
    """
    get the mode of a section of a histogram. Default params based on AoE values for correcting AoE
    """
    if cut==True:
        print(f'Using cut before finding mode: {cut_str}')
        df_plot = df.query(cut_str)
    else:
        df_plot = df
    hist, bins, vars = pgh.get_hist(df_plot[param], bins=a_bins, range=[alo, ahi])
    pars, cov = pgf.gauss_mode_width_max(hist, bins, vars)
    mode = pars[0]

    return(mode)

def get_superpulse(df, dg, cut_str='', nwfs = 100, all = False, norm = True):
    """Create a super-pulse from waveforms passing a cut. Waveforms are first baseline-subtracted.
    """
    if all==True:
        nwfs = len(df.query(cut_str))

        print(f'using all {nwfs} Waveforms passing cut')
    else:
        print(f'using first {nwfs} waveforms passing cut' )

    idx = df.query(cut_str).index[:nwfs]
    raw_store = lh5.Store()
    tb_name = 'ORSIS3302DecoderForEnergy/raw'
    lh5_dir = dg.lh5_dir
    raw_list = lh5_dir + dg.fileDB['raw_path'] + '/' + dg.fileDB['raw_file']
    raw_list = raw_list.tolist() # right now lh5.store.read_object() only works for lists, so need to convert the pandas object to a list first
    data_raw, nrows = raw_store.read_object(tb_name, raw_list)

    wfs_all = (data_raw['waveform']['values']).nda
    wfs = wfs_all[idx.values, :]
    # baseline subtraction
    bl_means = wfs[:,:3500].mean(axis=1)
    wf_blsub = (wfs.transpose() - bl_means).transpose()
    # ts = np.arange(0, wf_blsub.shape[1]-1, 1) # old way of doing it that Joule doesn't like-- makes the samles
                                                # and waveforms different lengths
    ts = np.arange(0, wf_blsub.shape[1], 1)
    super_wf = np.mean(wf_blsub, axis=0)
    wf_max = np.amax(super_wf)
    if norm == True:
        superpulse = np.divide(super_wf, wf_max)
    else:
        superpulse = super_wf
    return(ts, superpulse)



def get_superpulse_taligned(df, dg, cut_str='', nwfs = 100, all = False, norm = True,
                            tp_align=0.5, n_pre = 3800, n_post = 4000):
    """
    Create a super-pulse from waveforms passing a cut. Waveforms are first baseline-subtracted, Then time-aligned
    tp_align: the percentage of the max height to align WFs
    n_pre/_post: the window around the aligned timepoint value. Should be as large as possible if you want to keep most of the waveform.

    """
    if all==True:
        nwfs = len(df.query(cut_str))

        print(f'using all {nwfs} Waveforms passing cut')
    else:
        print(f'using first {nwfs} waveforms passing cut' )

    idx = df.query(cut_str).index[:nwfs]
    raw_store = lh5.Store()
    tb_name = 'ORSIS3302DecoderForEnergy/raw'
    lh5_dir = dg.lh5_dir
    raw_list = lh5_dir + dg.fileDB['raw_path'] + '/' + dg.fileDB['raw_file']
    raw_list = raw_list.tolist() # right now lh5.store.read_object() only works for lists, so need to convert the pandas object to a list first
    data_raw, nrows = raw_store.read_object(tb_name, raw_list)

    wfs_all = (data_raw['waveform']['values']).nda
    wfs = wfs_all[idx.values, :]
    # baseline subtraction
    bl_means = wfs[:,:3500].mean(axis=1)
    wf_blsub = (wfs.transpose() - bl_means).transpose()
    # ts = np.arange(0, wf_blsub.shape[1]-1, 1) # old way of doing it that Joule doesn't like-- makes the samles
                                                # and waveforms different lengths
    # ts = np.arange(0, wf_blsub.shape[1], 1)

    # time alignment
    wf_maxes = np.amax(wf_blsub, axis=1)
    timepoints = np.argmax(wf_blsub >= wf_maxes[:, None]*tp_align, axis=1)
    # timepoints = dsp.time_point_thresh(wf_blsub, wf_maxes[:, None]*tp_align, np.argmax(wf_blsub, axis=1), 0) # do this instead
                                                                                                            # when upgrade pygama
    # print(timepoints)
    wf_align = np.zeros([wf_blsub.shape[0], n_pre + n_post], dtype=wf_blsub.dtype)
    for i, tp in enumerate(timepoints):
        wf_align[i, :] = wf_blsub[i, tp-n_pre:tp+n_post]

    ts = np.arange(0, n_pre + n_post, 1)

    super_wf = np.mean(wf_align, axis=0)

    if norm == True:
        wf_max = np.amax(super_wf)
        superpulse = np.divide(super_wf, wf_max)
        return(ts, superpulse)
    else:
        return(ts, super_wf)


def get_superpulse_comp(df, dg, cut_str='', nwfs = 100, all = False, norm = True):
    """Create a super-pulse from waveforms passing a cut. Waveforms are first baseline-subtracted.
       Unlike `get_superpulse()`, this does not average waveforms
    """
    if all==True:
        nwfs = len(df.query(cut_str))

        print(f'using all {nwfs} Waveforms passing cut')
    else:
        print(f'using first {nwfs} waveforms passing cut' )

    idx = df.query(cut_str).index[:nwfs]
    raw_store = lh5.Store()
    tb_name = 'ORSIS3302DecoderForEnergy/raw'
    lh5_dir = dg.lh5_dir
    raw_list = lh5_dir + dg.fileDB['raw_path'] + '/' + dg.fileDB['raw_file']
    raw_list = raw_list.tolist() # right now lh5.store.read_object() only works for lists, so need to convert the pandas object to a list first
    data_raw, nrows = raw_store.read_object(tb_name, raw_list)

    wfs_all = (data_raw['waveform']['values']).nda
    wfs = wfs_all[idx.values, :]
    # baseline subtraction
    bl_means = wfs[:,:3500].mean(axis=1)
    wf_blsub = (wfs.transpose() - bl_means).transpose()
    # ts = np.arange(0, wf_blsub.shape[1]-1, 1) # old way of doing it that Joule doesn't like-- makes the samles
                                                # and waveforms different lengths
    ts = np.arange(0, wf_blsub.shape[1], 1)
    super_wf = np.sum(wf_blsub, axis=0)
    superpulse = super_wf
    return(ts, superpulse)

def get_superpulse_window(df, dg, cut_str=[''], nwfs = 100, all = False, norm = True):
    """Create a super-pulse from waveforms passing a cut. Waveforms are first baseline-subtracted.
    """
    waveforms = []
    energies = []
    for cut in cut_str:
        print(str(cut))

        if all==True:
            nwfs = len(df.query(str(cut)).copy())

            print(f'using all {nwfs} Waveforms passing cut')
        else:
            print(f'using first {nwfs} waveforms passing cut' )

        idx = df.query(str(cut)).copy().index[:nwfs]
        raw_store = lh5.Store()
        tb_name = 'ORSIS3302DecoderForEnergy/raw'
        lh5_dir = dg.lh5_dir
        raw_list = lh5_dir + dg.fileDB['raw_path'] + '/' + dg.fileDB['raw_file']
        raw_list = raw_list.tolist() # right now lh5.store.read_object() only works for lists, so need to convert the pandas object to a list first
        data_raw, nrows = raw_store.read_object(tb_name, raw_list)

        wfs_all = (data_raw['waveform']['values']).nda
        wfs = wfs_all[idx.values, :]
        # baseline subtraction
        bl_means = wfs[:,:3500].mean(axis=1)
        wf_blsub = (wfs.transpose() - bl_means).transpose()
        ts = np.arange(0, wf_blsub.shape[1]-1, 1)
        super_wf = np.mean(wf_blsub, axis=0)
        wf_max = np.amax(super_wf)
        if norm == True:
            superpulse = np.divide(super_wf, wf_max)
        else:
            superpulse = super_wf
        waveforms.append(superpulse)
    waveforms = np.asarray(waveforms)

    super_duper_wf = np.mean(wf_blsub, axis=0)
    wf_super_max = np.amax(super_duper_wf)
    if norm == True:
        super_duper_pulse = np.divide(super_duper_wf, wf_super_max)
    else:
        super_duper_pulse = super_duper_wf
    return(ts, waveforms, super_duper_pulse)

def get_wfs(df, dg, cut_str='', nwfs = 10, all = False, norm=False, tp_align=0.5, n_pre = 3800, n_post = 4000):
    """Get waveforms passing a cut, baseline-subtracted but not normalized. These are individual waveforms, not superpulses!

    time-aligned
    """
    all_nwfs = len(df.query(cut_str).copy())
    print(f'{all_nwfs} passing cuts')

    if all==True:
        nwfs = len(df.query(cut_str).copy())

        print(f'using all {nwfs} Waveforms passing cut')

    else:
        print(f'using first {nwfs} waveforms passing cut' )

    if all_nwfs < nwfs:
        print(f'Less than the specified number of waveforms ({nwfs}) passing cuts. \nUsing all {all_nwfs} waveforms passing cut')
        nwfs = all_nwfs

    idx = df.query(cut_str).copy().index[:nwfs]
    raw_store = lh5.Store()
    tb_name = 'ORSIS3302DecoderForEnergy/raw'
    lh5_dir = dg.lh5_dir
    raw_list = lh5_dir + dg.fileDB['raw_path'] + '/' + dg.fileDB['raw_file']
    raw_list = raw_list.tolist() # right now lh5.store.read_object() only works for lists, so need to convert the pandas object to a list first
    data_raw, nrows = raw_store.read_object(tb_name, raw_list)

    wfs_all = (data_raw['waveform']['values']).nda
    wfs = wfs_all[idx.values, :]
    # baseline subtraction
    bl_means = wfs[:,:3500].mean(axis=1)
    wf_blsub = (wfs.transpose() - bl_means).transpose()

    wf_maxes = np.amax(wf_blsub, axis=1)
    timepoints = np.argmax(wf_blsub >= wf_maxes[:, None]*tp_align, axis=1)
    # timepoints = dsp.time_point_thresh(wf_blsub, wf_maxes[:, None]*tp_align, np.argmax(wf_blsub, axis=1), 0) # do this instead
                                                                                                            # when upgrade pygama
    # print(timepoints)
    wf_align = np.zeros([wf_blsub.shape[0], n_pre + n_post], dtype=wf_blsub.dtype)
    for i, tp in enumerate(timepoints):
        wf_align[i, :] = wf_blsub[i, tp-n_pre:tp+n_post]

    ts = np.arange(0, n_pre + n_post, 1)

    if norm == True:
        wf_max = np.amax(wf_align)
        wf_norm = np.divide(wf_align, wf_max)
        return(ts, wf_norm)

    else:
        return(ts, wf_align)

def double_pole_zero(wf_in, tau1, tau2, frac):
    """
    Pole-zero correction an array of input waveforms using two time constants: one main (long) time constant
    tau1, and a shorter time constant tau2 that contributes a fraction frac
    wf_in: array of input waveforms
    tau1, tau2: time constants in sample equivalents
    adapted from pygama/dsp/_processors
    """
    const1 = 1/tau1 #np.exp(-1/tau1)
    const2 = 1/tau2 #np.exp(-1/tau2)
    wf_out_arr = []
    print(f'pole-zero correcting {len(wf_in)} waveforms')
    for n in range(len(wf_in)):
        wf_out = np.zeros(len(wf_in[n]))
        wf_out[0] = wf_in[n][0]
        e1 = wf_in[n][0]
        e2 = wf_in[n][0]
        e3 = 0
        for i in range(1, len(wf_in[n])):
            e1 += wf_in[n][i] - e2 + e2*const1
            e3 += wf_in[n][i] - e2 - e3*const2
            e2 = wf_in[n][i]
            wf_out[i] = e1 - frac*e3
        wf_out_arr.append(wf_out)

    return(wf_out_arr)

def notchFilter(waveform, f_notch, Q):
    """
    apply notch filter of some frequency f_notch (Hz) with some quality factor Q
    """
    wf = waveform
    clk = 100e6 # 100 MHz sampling frequency of SIS 3302

    b_notch, a_notch = signal.iirnotch(f_notch, Q, clk)
    wf_notch = signal.filtfilt(b_notch, a_notch, wf)
    return wf_notch

def notchFilter_SIS3302(waveform, Q):
    """
    specific notch filter to get rid of 25 MHz and 50 MHz noise in the SIS 3302 used by CAGE.
    Can specify the quality factor
    """
    wf = waveform
    f_notch1 = 25e6
    f_notch2 = 50e6

    pre_notch = notchFilter(wf, f_notch1, Q)
    wf_notch = notchFilter(pre_notch, f_notch2, Q)


    return wf_notch

def peakCounts(df, energy_par='trapEftp_cal', bins=50, erange=[], bkg_sub=True, writeParams=False):
    """
    Get the number of counts in a peak. Can be sideband-subtracted or raw.
    Recommend getting pgfenergy_hist, pgfebins, evars using pgh.get_hist()
    """

    if len(erange) < 2:
        print('Must specify an energy range for the fit!')
        exit()

    # First use gauss_mode_width_max to use for initial guesses in fit_hist
    ehist, ebins, evars = pgh.get_hist(df[energy_par], bins=bins, range=erange)
    pars, cov = pgf.gauss_mode_width_max(ehist, ebins, evars)
    mode = pars[0]
    width = pars[1]
    amp = pars[2]
    print('Guess: {pars}')
    # print(f'mode: {mode}')
    # print(f'width: {width}')
    # print(f'amp: {amp}')

    e_pars, ecov = pgf.fit_hist(gauss_fit_func, ehist, ebins, evars, guess = (amp, mode, width, 1))

    chi_2 = pgf.goodness_of_fit(ehist, ebins, gauss_fit_func, e_pars)

    mean = e_pars[1]
    mean_err = ecov[1]

    sig = e_pars[2]
    sig_err = ecov[2]

    en_amp_fit = e_pars[0]
    en_const_fit = e_pars[3]

    fwhm = sig*2.355

    print(f'chi square: {chi_2}')
    print(f'mean: {mean}')
    print(f'width: {sig}')
    print(f'amp: {en_amp_fit}')
    print(f'C: {en_const_fit}')
    print(f'FWHM: {fwhm} \n{(fwhm/mean)*100}%')

    cut_3sig = f'({mean-3*sig} <= {energy_par} <= {mean+3*sig})'

    counts_peak = len(df.query(cut_3sig).copy())
    err_peak = np.sqrt(counts_peak)

    print(f'peak counts: {counts_peak}')
    print(f'error: {err_peak}')

    if bkg_sub==True:
        bkg_left_min = mean-7.*sig
        bkg_left_max = mean-4*sig

        bkg_right_min = mean+4*sig
        bkg_right_max = mean+7.*sig

        bkg_left = f'({bkg_left_min} <= {energy_par} < {bkg_left_max})'
        bkg_right = f'({bkg_right_min} < {energy_par} <= {bkg_right_max})'

        bkg = f'{bkg_left} or {bkg_right}'

        left_counts = len(df.query(bkg_left).copy())
        right_counts = len(df.query(bkg_right).copy())
        total_bkg = left_counts + right_counts
        err_bkg = np.sqrt(total_bkg)


        bkg_sub_counts = counts_peak - total_bkg
        err = np.sqrt(counts_peak + total_bkg)

        print(f'peak counts: {counts_peak}')
        print(f'bkg left: {left_counts}')
        print(f'bkg right: {right_counts}')
        print(f'total bkg: {total_bkg}')

        print(f'bkg_subtracted counts: {bkg_sub_counts}')
        print(f'error: {err}')
        print(f'{(err/bkg_sub_counts)*100:.3f}%')

        return(bkg_sub_counts, err)

    else:
        return(counts_peak, err_peak)

def writeCuts(run, cut_key, cut):
    """
    Write cut string to cuts.json file.
    Cut should be in a string format that can then be passed to and interpreted by df.query() correctly for application of
    cuts!
    """
    print(f'Writing cut {cut_key}: {cut} to $CAGE_SW/analysis/cuts.json')
    with open('./cuts.json', 'r+') as f:
        cuts = json.load(f)
        if str(run) not in cuts:
            cuts[str(run)] = {}
        cuts[str(run)][cut_key] = cut
        f.seek(0)  # <--- should reset file position to the beginning.
        json.dump(cuts, f, indent=4, sort_keys=True) # <--- pretty printing of json file
        f.truncate()

def writeJson(file, run, param_key, param):
    """
    Write useful parameters, sorted by run number, to a specified json file for use in the future.
    """
    print(f'Writing parameter {param_key}: {param} to $CAGE_SW/analysis/cuts.json')
    with open(file, 'r+') as f:
        params = json.load(f)
        if str(run) not in params:
            params[str(run)] = {}
        params[str(run)][param_key] = param
        f.seek(0)  # <--- should reset file position to the beginning.
        json.dump(params, f, indent=4, sort_keys=True) # <--- pretty printing of json file
        f.truncate()

def gauss_fit_func(x, A, mu, sigma, C):
    """
    generic gaussian to use in basic fits
    """
    return (A * (1/(sigma*np.sqrt(2*np.pi))) *(np.exp(-1.0 * ((x - mu)**2) / (2 * sigma**2)))+C)

def gauss_2D(x, y, A=1., mu_x=0., mu_y=0., sigma_x=1., sigma_y=1., rot=0. ):
    """
    generic gaussian to use in 2D gaussian fits
    rot is the rotation angle of the gaussian
    """
    a = (np.cos(rot)**2/(2*sigma_x**2)) + (np.sin(rot)**2/(2*sigma_y**2))
    b = (-1*np.sin(2*rot)/(4*sigma_x**2) )+ (np.sin(2*rot)/(4*sigma_y**2))
    c = (np.sin(rot)**2/(2*sigma_x**2)) + (np.cos(rot)**2/(2*sigma_y**2))

    z = A*np.exp(-(a*(x-mu_x)**2 +2*b*((x-mu_x)*(y-mu_y)) +c*(y-mu_y)**2))

    return z


def time_point_thresh_max(wf_in, threshold, tp_max, max):
    """
    adapted from pygama DSP to use on single WFs
    Find the last timepoint before tp_max that wf_in crosses a threshold
     wf_in: input waveform
     threshold: threshold to search for
     tp_out: final time that waveform is less than threshold
    """
    tp_out = 0
    for i in range(tp_max, max, -1):
        if(wf_in[i]>threshold and wf_in[i-1]<threshold):
            tp_out = i

    return tp_out

def asymTrapFilter(wf_in, rise, flat, fall):
    """
    adapted from pygama DSP to use on single WFs
    Computes an asymmetric trapezoidal filter
    """
    wf_out = np.zeros(len(wf_in))
    wf_out[0] = wf_in[0]/float(rise)
    for i in range(1, rise):
        wf_out[i] = wf_out[i-1] + (wf_in[i])/float(rise)
    for i in range(rise, rise+flat):
        wf_out[i] = wf_out[i-1] + (wf_in[i] - wf_in[i-rise])/float(rise)
    for i in range(rise+flat, rise+flat+fall):
        wf_out[i] = wf_out[i-1] + (wf_in[i] - wf_in[i-rise])/float(rise) - wf_in[i-rise-flat]/float(fall)
    for i in range(rise+flat+fall, len(wf_in)):
        wf_out[i] = wf_out[i-1] + (wf_in[i] - wf_in[i-rise])/float(rise) - (wf_in[i-rise-flat] - wf_in[i-rise-flat-fall])/float(fall)

    return(wf_out)


def time_point_thresh(wf_in, threshold, tp_max):
    """
    adapted from pygama DSP to use on single WFs
    Find the last timepoint before tp_max that wf_in crosses a threshold
     wf_in: input waveform
     threshold: threshold to search for
     tp_out: final time that waveform is less than threshold
    """
    tp_out = 0
    for i in range(tp_max, 0, -1):
        if(wf_in[i]>threshold and wf_in[i-1]<threshold):
            tp_out = i
    return tp_out

def time_point_frac(wf_in, frac, tp_max):
    """
    adapted from pygama DSP to use on single WFs
    Find the time where the waveform crosses a value that is a fraction of the
    max. Parameters are:
     wf_in: input waveform. Should have baseline of 0!
     frac: fraction of maximum to search for. Should be between 0 and 1.
     tp_max: timepoint of wf maximum. Can be found with numpy.argmax
     tp_out: time that waveform crosses frac * wf_in[tp_max] for the last time,
     rounded to an integer index
    """
    # tp_out = [0]
    threshold = frac*wf_in[tp_max]
    # tp_out[0] = tp_max-1
    tp_out = tp_max-1
    # Scan for the crossing
    while(wf_in[tp_out] > threshold):
        tp_out -= 1
    # if the previous point is closer to the threshold than the one we landed on
    # use that. This is equivalent to interpolating and then rounding
    if(threshold - wf_in[tp_out] > wf_in[tp_out+1] - threshold):
        tp_out += 1
    return tp_out

def trap_norm(wf_in, rise, flat):
    """
    adapted from pygama DSP to use on single WFs
    Symmetric trapezoidal filter normalized by integration time
    """
    wf_out = np.zeros(len(wf_in))
    wf_out[0] = wf_in[0]/float(rise)
    for i in range(1, rise):
        wf_out[i] = wf_out[i-1] + wf_in[i]/float(rise)
    for i in range(rise, rise+flat):
        wf_out[i] = wf_out[i-1] + (wf_in[i] - wf_in[i-rise])/float(rise)
    for i in range(rise+flat, 2*rise+flat):
        wf_out[i] = wf_out[i-1] + (wf_in[i] - wf_in[i-rise] - wf_in[i-rise-flat])/float(rise)
    for i in range(2*rise+flat, len(wf_in)):
        wf_out[i] = wf_out[i-1] + (wf_in[i] - wf_in[i-rise] - wf_in[i-rise-flat] + wf_in[i-2*rise-flat])/float(rise)

    return wf_out


def testFunc(list):
    if 'test' in list:
        print("hi it worked")
