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
import scipy.optimize as opt

from pygama import DataGroup
import pygama.lh5 as lh5
import pygama.analysis.histograms as pgh
import pygama.analysis.peak_fitting as pgf
import cage_utils

mpl.use('Agg')

def main():
    # runs = [38]
    # runs = [38, 60, 42, 64, 44, 66, 48, 70, 50, 72, 54]
    # runs = [120, 121, 123, 124, 126, 128, 129, 131, 132, 134, 135, 137, 143]
    run_list = [60, 64, 66, 70, 72] # alpha runs for dsp_id = 2
#     runs = [62, 68, 74] #bkg runs for dsp_id = 2
#     runs = [50]
#     alp_runs = [137, 143]
#     bkg_runs = [136, 136]
    # campaign = 'angleScan/'
    campaign = 'new_normScan/'
    
    run=72

    user = True
    hit = True
    cal = True
    lowE = False
    etype = 'trapEftp_cal'

    dsp_list = ['energy', 'trapEftp', 'trapEmax', 'trapEftp_cal', 'bl','bl_sig', 'bl_slope', 'AoE', 'dcr', "tp_0",
            "tp_02", "tp_05", "tp_10", "tp_20", 'tp_max', 'ToE', 'log_tail_fit_slope', 'wf_max', 'wf_argmax', 'trapE_argmax', 'lf_max']

    cut_keys = set(['wf_max_cut', 'bl_mean_cut_raw', 'bl_mean_cut', 'bl_slope_cut_raw', 'bl_slope_cut',
            'bl_sig_cut_raw', 'bl_sig_cut', 'ftp_max_cut_raw', 'ftp_max_cut'])
    
    cut_keys_raw = set(['wf_max_cut', 'bl_mean_cut_raw', 'bl_slope_cut_raw',
            'bl_sig_cut_raw', 'ftp_max_cut_raw'])


    df_raw, dg, runtype, rt_min, radius, angle_det, rotary = cage_utils.getDataFrame(run, user=user, hit=hit, cal=cal, dsp_list=dsp_list, lowE=lowE)
    
    df_raw['ftp_max'] = df_raw['trapEftp']/df_raw['trapEmax']
    
    df = cage_utils.apply_DC_Cuts(run, df_raw)
    
    rateVSrad(run_list, dsp_list, user=True, hit=True, cal=True, lowE=False)

    # peakCounts_60(run, df, runtype, rt_min, radius, angle_det, rotary, energy_par=etype, bins=50, erange=[54,65], bkg_sub=True, plot=True, writeParams=True)

    
def rateVSrad(run_list, dsp_list, user=True, hit=True, cal=True, lowE=False):
    
    rad_arr = []
    counts_arr = []
    err_arr = []
    
    for run in run_list:
        df_raw, dg, runtype, rt_min, radius, angle_det, rotary = cage_utils.getDataFrame(run, user=user, hit=hit, cal=cal, dsp_list=dsp_list, lowE=lowE)
    
        df = cage_utils.apply_DC_Cuts(run, df_raw)
        
        counts, err = peakCounts_60(run, df, runtype, rt_min, radius, angle_det, rotary, energy_par='trapEftp_cal', bins=50, erange=[54,65], bkg_sub=True, plot=False, writeParams=False)
                                      
        counts_arr.append(counts)
        err_arr.append(err)
        rad_arr.append(radius)
    
    fig, ax = plt.subplots()
    
    plt.errorbar(rad_arr, counts_arr, yerr=err_arr, marker = '.', c='b', ls='none', label=f'counts')
    
    plt.xlabel('Radial Position (mm)', fontsize=14)
    plt.ylabel('Counts', fontsize=16)
    plt.title(f'60 keV Counts VS Radius', fontsize=14)
    plt.setp(ax.get_xticklabels(), fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)
#     plt.legend()
    plt.tight_layout()
    plt.savefig(f'./plots/new_normScan/60keV_analysis/rate_60keV_vs_rad.png', dpi=200)
    
    plt.clf()
    plt.close()

def peakCounts_60(run, df, runtype, rt_min, radius, angle_det, rotary, energy_par='trapEftp_cal', bins=50, erange=[54,65], bkg_sub=True, plot=False, writeParams=False):
    """
    Get the number of counts in the 60 keV peak, make plots. Can be sideband-subtracted or raw.
    Taken partially from cage_utils.py, adapted to be specific for 60 keV analysis
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
    print(f'Guess: {pars}')
    # print(f'mode: {mode}')
    # print(f'width: {width}')
    # print(f'amp: {amp}')
    
    e_pars, ecov = pgf.fit_hist(cage_utils.gauss_fit_func, ehist, ebins, evars, guess = (amp, mode, width, 1))

    chi_2 = pgf.goodness_of_fit(ehist, ebins, cage_utils.gauss_fit_func, e_pars)

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
    
    if plot==True:
        fig, ax = plt.subplots()

        plt.plot(ebins[1:], cage_utils.gauss_fit_func(ebins[1:], *e_pars), c = 'r', lw=0.8, label='gaussian fit')
        plt.plot(ebins[1:], ehist, ds='steps', c='b', lw=1.)
        
        plt.axvline(mean-3*sig, c='g', lw=1, label ='Peak region (3 sigma)')
        plt.axvline(mean+3*sig, c='g', lw=1)

        plt.xlabel('Energy (keV)', fontsize=14)
        plt.ylabel('counts', fontsize=14)

        plt.title(f'60 keV peak with gaussian fit', fontsize = 14)

        plt.setp(ax.get_xticklabels(), fontsize=12)
        plt.setp(ax.get_yticklabels(), fontsize=12)

        ax.text(0.03, 0.8,  f'r = {radius} mm \ntheta = {angle_det} deg \nruntime {rt_min:.2f}',
                verticalalignment='bottom',horizontalalignment='left', transform=ax.transAxes, color='black', 
                fontsize=10, bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 8})
        ax.text(0.95, 0.8,  f'mean: {mean:.2f} \nsigma: {sig:.3f} \nchi square: {chi_2:.2f}', 
                verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes, color='black',
                fontsize=10, bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 8})
        
        plt.legend(loc='center right')
        
        plt.tight_layout()

        plt.savefig(f'./plots/new_normScan/60keV_analysis/run{run}_fit_60keV.png', dpi=200)
        plt.clf()
        plt.close()
    
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
        
        if plot==True:
            fig, ax = plt.subplots()
            
            full_hist,  full_bins, full_evars = pgh.get_hist(df[{energy_par}], bins=70, range=[mean-9.*sig,
                                                                                                 mean+9.*sig])

            plt.plot(full_bins[1:], full_hist, ds='steps', c='b', lw=1)
            
            # plt.axvline(mean-3*sig, c='g', lw=1, label ='Peak region')
            # plt.axvline(mean+3*sig, c='g', lw=1)
            
            ax.axvspan(mean-3*sig, mean+3*sig, alpha=0.1, color='g', label='peak region (3 sigma)')

            # plt.axvline(bkg_left_min, c='r', lw=1, label='Background region')
            # plt.axvline(bkg_left_max, c='r', lw=1)

            # plt.axvline(bkg_right_min, c='r', lw=1)
            # plt.axvline(bkg_right_max, c='r', lw=1)
            
            ax.axvspan(bkg_left_min, bkg_left_max, alpha=0.2, color='r', label='background region (3 sigma)')
            ax.axvspan(bkg_right_min, bkg_right_max, alpha=0.2, color='r')
            
            plt.title('60 keV peak with background subtraction region', fontsize=14)
            
            plt.xlabel(f'{energy_par} (keV)', fontsize=14)
            plt.ylabel('counts', fontsize=14)
            

            plt.setp(ax.get_xticklabels(), fontsize=12)
            plt.setp(ax.get_yticklabels(), fontsize=12)
            
            ax.text(0.03, 0.8,  f'r = {radius} mm \ntheta = {angle_det} deg \nruntime {rt_min:.2f}',
                verticalalignment='bottom',horizontalalignment='left', transform=ax.transAxes, color='black', 
                fontsize=10, bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 8})
            
            plt.legend(loc='upper right')
            
            plt.tight_layout()
            
            plt.savefig(f'./plots/new_normScan/60keV_analysis/run{run}_bkgRegion_60keV.png', dpi=200)
            
            plt.clf()
            plt.close()
        
        # For Joule's 60keV analysis. Generally don't do this
        if writeParams==True:
            param_keys = ['mean_60', 'sig_60', 'chiSquare_fit_60', 'cut_60_3sig','bkg_60_left',
                          'bkg_60_right', 'bkg_60']
            param_list = [mean, sig, chi_2, cut_3sig, bkg_left, bkg_right, bkg]
            
            for key, cut in zip(param_keys, param_list):
                cage_utils.writeJson('./analysis_60keV.json', run, key, cut)
        
        return(bkg_sub_counts, err)
    
    else:
        return(counts_peak, err_peak)


if __name__=="__main__":
    main()
