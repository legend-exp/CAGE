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
    # run_list = [60, 64]
    run_list = [60, 64, 66, 70, 72] # alpha runs for dsp_id = 2
    # run_list = [89, 91, 93, 95] # centering alpha runs for dsp_id = 1 and 2
    # runs = [62, 68, 74] #bkg runs for dsp_id = 2
    # campaign = 'angleScan/'
    campaign = 'new_normScan/60keV_analysis/'
    # campaign = 'new_normScan/centeringScans/'
    
    # run=95

    user = True
    hit = True
    cal = True
    lowE = False
    etype = 'trapEftp_cal'

    dsp_list = ['energy', 'trapEftp', 'trapEmax', 'trapEftp_cal', 'bl','bl_sig', 'bl_slope', 'AoE', 'dcr', "tp_0",
                "tp_02", "tp_05", "tp_10", "tp_20", 'tp_30', 'tp_50', 'tp_90', 'tp_max', 'ToE', 'log_tail_fit_slope', 
                'wf_max', 'wf_argmax', 'trapE_argmax', 'lf_max']

    cut_keys = set(['wf_max_cut', 'bl_mean_cut_raw', 'bl_mean_cut', 'bl_slope_cut_raw', 'bl_slope_cut',
            'bl_sig_cut_raw', 'bl_sig_cut', 'ftp_max_cut_raw', 'ftp_max_cut'])
    
    cut_keys_raw = set(['wf_max_cut', 'bl_mean_cut_raw', 'bl_slope_cut_raw',
            'bl_sig_cut_raw', 'ftp_max_cut_raw'])
    
    # for run in run_list:
        # peakCounts_60(run, campaign, user=user, hit=hit, cal=cal, dsp_list=dsp_list, lowE=lowE, energy_par=etype, bins=30, erange=[54,65], bkg_sub=True, plot=True, writeParams=False)

    
    rateVSrad(run_list, campaign, dsp_list, norm=True, user=True, hit=True, cal=True, lowE=False)
    
    # superpulses(run_list, campaign, dsp_list, norm = False, user=True, hit=True, cal=True, lowE=False, 
                # write_data=True)
    
    # risetime_dists(run_list, campaign, dsp_list, norm = False, user=True, hit=True, cal=True, lowE=False,
                   # savefig=True, writeParams=False)
    
    # risetime_superpulses(run_list, campaign, dsp_list, norm = False, user=True, hit=True, cal=True,
                         # lowE=False, savefig=True)
    # plot_superpulses(campaign, savefig=True, inset=False)
    
    # plot_bkgSub_superpulses(campaign, savefig=True)
    
        

    
def rateVSrad(run_list, campaign, dsp_list, norm = False, user=True, hit=True, cal=True, lowE=False):
    """
    Get and plot the rate of 60 keV gammas after cuts, versus radius
    """
    
    rad_arr = []
    counts_arr = []
    err_arr = []
    
    for run in run_list:
        df_raw, dg, runtype, rt_min, radius, angle_det, rotary = cage_utils.getDataFrame(run, user=user, hit=hit,
                                                                                         cal=cal, dsp_list=dsp_list,
                                                                                         lowE=lowE)
    
        df = cage_utils.apply_DC_Cuts(run, df_raw)
        
        counts, err = peakCounts_60(run, campaign, dsp_list, user=user, hit=hit, cal=cal, lowE=lowE,
                  energy_par='trapEftp_cal', bins=30, erange=[54,65], bkg_sub=True, plot=False, 
                  writeParams=False)
            
        if norm==True:
            counts_arr.append(counts/rt_min)
            err_arr.append(err/rt_min)
            
        else:
            counts_arr.append(counts)
            err_arr.append(err)
            
        rad_arr.append(radius)
        
            
    
    fig, ax = plt.subplots()
    
    plt.errorbar(rad_arr, counts_arr, yerr=err_arr, marker = 'o', c='b', ls='none', label=f'counts')
    
    plt.xlabel('Radial Position (mm)', fontsize=14)
    if norm==True:
        plt.ylabel('Counts/min', fontsize=14)
    else:
        plt.ylabel('Counts', fontsize=14)
    plt.title(f'60 keV Counts VS Radius', fontsize=14)
    plt.setp(ax.get_xticklabels(), fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)
    # plt.legend()
    plt.tight_layout()
    plt.savefig(f'./plots/{campaign}rate_60keV_vs_rad.png', dpi=200)
    plt.savefig(f'./plots/{campaign}rate_60keV_vs_rad.pdf', dpi=200)
    
    plt.clf()
    plt.close()
    

        

def peakCounts_60(run, campaign, dsp_list, user=True, hit=True, cal=True, lowE=False,
                  energy_par='trapEftp_cal', bins=50, erange=[54,65], bkg_sub=True, plot=False, 
                  writeParams=False):
    """
    Get the number of counts in the 60 keV peak, make plots. Can be sideband-subtracted or raw.
    Taken partially from cage_utils.py, adapted to be specific for 60 keV analysis
    """
    
    if len(erange) < 2:
        print('Must specify an energy range for the fit!')
        exit()
        
    df_raw, dg, runtype, rt_min, radius, angle_det, rotary = cage_utils.getDataFrame(run, user=user, hit=hit,cal=cal,
                                                                                     dsp_list=dsp_list, lowE=lowE)
    
    df = cage_utils.apply_DC_Cuts(run, df_raw)
    
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
    
    # Standard 3 sigma cut for rate analysis
    cut_3sig = f'({mean-3*sig} <= {energy_par} <= {mean+3*sig})'    
    counts_peak = len(df.query(cut_3sig).copy())
    err_peak = np.sqrt(counts_peak)
    
    # Also make cuts for 2 and 3 sigma for systematic studies with superpulses
    cut_2sig = f'({mean-2*sig} <= {energy_par} <= {mean+2*sig})'
    cut_1sig = f'({mean-sig} <= {energy_par} <= {mean+sig})'
    counts_peak_2sig = len(df.query(cut_2sig).copy())
    counts_peak_1sig = len(df.query(cut_1sig).copy())
    err_peak_2sig = np.sqrt(counts_peak_2sig)
    err_peak_1sig = np.sqrt(counts_peak_1sig)

    
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

        ax.text(0.03, 0.8,  f'r = {radius} mm \ntheta = {angle_det} deg \nruntime {rt_min:.2f} min',
                verticalalignment='bottom',horizontalalignment='left', transform=ax.transAxes, color='black', 
                fontsize=10, bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 8})
        ax.text(0.95, 0.8,  f'mean: {mean:.2f} keV \nsigma: {sig:.3f} keV \nchi square: {chi_2:.2f}', 
                verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes, color='black',
                fontsize=10, bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 8})
        
        plt.legend(loc='center right')
        
        plt.tight_layout()

        plt.savefig(f'./plots/{campaign}run{run}_fit_60keV.png', dpi=200)
        plt.savefig(f'./plots/{campaign}run{run}_fit_60keV.pdf', dpi=200)
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
        
        # determine 1 and 2 sigma background cuts for systematic studies with superpulses
        bkg_left_min_1sig = mean-5.*sig
        bkg_left_max_1sig = mean-4*sig

        bkg_right_min_1sig = mean+4*sig
        bkg_right_max_1sig = mean+5.*sig

        bkg_left_1sig = f'({bkg_left_min_1sig} <= {energy_par} < {bkg_left_max_1sig})'
        bkg_right_1sig = f'({bkg_right_min_1sig} < {energy_par} <= {bkg_right_max_1sig})'
        
        bkg_1sig = f'{bkg_left_1sig} or {bkg_right_1sig}'
        
        left_counts_1sig = len(df.query(bkg_left_1sig).copy())
        right_counts_1sig = len(df.query(bkg_right_1sig).copy())
        total_bkg_1sig = left_counts_1sig + right_counts_1sig
        err_bkg_1sig = np.sqrt(total_bkg_1sig)
        
        bkg_left_min_2sig = mean-6.*sig
        bkg_left_max_2sig = mean-4*sig

        bkg_right_min_2sig = mean+4*sig
        bkg_right_max_2sig = mean+6.*sig

        bkg_left_2sig = f'({bkg_left_min_2sig} <= {energy_par} < {bkg_left_max_2sig})'
        bkg_right_2sig = f'({bkg_right_min_2sig} < {energy_par} <= {bkg_right_max_2sig})'
        
        bkg_2sig = f'{bkg_left_2sig} or {bkg_right_2sig}'
        
        left_counts_2sig = len(df.query(bkg_left_2sig).copy())
        right_counts_2sig = len(df.query(bkg_right_2sig).copy())
        total_bkg_2sig = left_counts_2sig + right_counts_2sig
        err_bkg_2sig = np.sqrt(total_bkg_2sig)


        if plot==True:
            fig, ax = plt.subplots()
            
            full_hist,  full_bins, full_evars = pgh.get_hist(df[{energy_par}], bins=bins, range=[mean-9.*sig,
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
            
            ax.text(0.03, 0.8,  f'r = {radius} mm \ntheta = {angle_det} deg \nruntime {rt_min:.2f} min',
                verticalalignment='bottom',horizontalalignment='left', transform=ax.transAxes, color='black', 
                fontsize=10, bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 8})
            
            plt.legend(loc='upper right')
            
            plt.tight_layout()
            
            plt.savefig(f'./plots/{campaign}run{run}_bkgRegion_60keV.png', dpi=200)
            plt.savefig(f'./plots/{campaign}run{run}_bkgRegion_60keV.pdf', dpi=200)
            
            plt.clf()
            plt.close()
            
            # for 2 sigma region
            fig, ax = plt.subplots()

            plt.plot(full_bins[1:], full_hist, ds='steps', c='b', lw=1)
            
            # plt.axvline(mean-3*sig, c='g', lw=1, label ='Peak region')
            # plt.axvline(mean+3*sig, c='g', lw=1)
            
            ax.axvspan(mean-2*sig, mean+2*sig, alpha=0.1, color='g', label='peak region (2 sigma)')

            # plt.axvline(bkg_left_min, c='r', lw=1, label='Background region')
            # plt.axvline(bkg_left_max, c='r', lw=1)

            # plt.axvline(bkg_right_min, c='r', lw=1)
            # plt.axvline(bkg_right_max, c='r', lw=1)
            
            ax.axvspan(bkg_left_min_2sig, bkg_left_max_2sig, alpha=0.2, color='r', label='background region (2 sigma)')
            ax.axvspan(bkg_right_min_2sig, bkg_right_max_2sig, alpha=0.2, color='r')
            
            plt.title('60 keV peak with background subtraction region', fontsize=14)
            
            plt.xlabel(f'{energy_par} (keV)', fontsize=14)
            plt.ylabel('counts', fontsize=14)
            

            plt.setp(ax.get_xticklabels(), fontsize=12)
            plt.setp(ax.get_yticklabels(), fontsize=12)
            
            ax.text(0.03, 0.8,  f'r = {radius} mm \ntheta = {angle_det} deg \nruntime {rt_min:.2f} min',
                verticalalignment='bottom',horizontalalignment='left', transform=ax.transAxes, color='black', 
                fontsize=10, bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 8})
            
            plt.legend(loc='upper right')
            
            plt.tight_layout()
            
            plt.savefig(f'./plots/{campaign}run{run}_bkgRegion_2sigma_60keV.png', dpi=200)
            plt.savefig(f'./plots/{campaign}run{run}_bkgRegion_2sigma_60keV.pdf', dpi=200)
            
            plt.clf()
            plt.close()

        # For Joule's 60keV analysis. Generally don't do this
        if writeParams==True:
            param_keys = ['mean_60', 'sig_60', 'chiSquare_fit_60', 'cut_60_3sig','bkg_60_left',
                          'bkg_60_right', 'bkg_60', 'counts_peak', 'err_peak', 'counts_bkg', 'err_bkg',
                          'cut_60_2sig','bkg_60_left_2sig','bkg_60_right_2sig', 'bkg_60_2sig', 'counts_peak_2sig',
                          'err_peak_2sig', 'counts_bkg_2sig', 'err_bkg_2sig', 'cut_60_1sig','bkg_60_left_1sig',
                          'bkg_60_right_1sig', 'bkg_60_1sig', 'counts_peak_1sig', 'err_peak_1sig', 'counts_bkg_1sig',
                          'err_bkg_1sig']
            param_list = [mean, sig, chi_2, cut_3sig, bkg_left, bkg_right, bkg, counts_peak, err_peak, total_bkg,
                          err_bkg, cut_2sig, bkg_left_2sig, bkg_right_2sig, bkg_2sig, counts_peak_2sig,
                          err_peak_2sig, total_bkg_2sig, err_bkg_2sig, cut_1sig, bkg_left_1sig, bkg_right_1sig,
                          bkg_1sig, counts_peak_1sig, err_peak_1sig, total_bkg_1sig, err_bkg_1sig]
            
            for key, cut in zip(param_keys, param_list):
                cage_utils.writeJson('./analysis_60keV.json', run, key, cut)
        
        return(bkg_sub_counts, err)
    
    else:
        return(counts_peak, err_peak)
    
def superpulses(run_list, campaign, dsp_list, norm = False, user=True, hit=True, cal=True, lowE=False,
                write_data=True):
    # open dataframe with 60 keV signal and background region info
    with open('./analysis_60keV.json') as f:
        params = json.load(f)
        
    for (run, idx) in zip(run_list, np.arange(len(run_list))):
        df_raw, dg, runtype, rt_min, radius, angle_det, rotary = cage_utils.getDataFrame(run, user=user, hit=hit,
                                                                                         cal=cal, dsp_list=dsp_list,
                                                                                         lowE=lowE)
    
        df = cage_utils.apply_DC_Cuts(run, df_raw)
        
        mean_60 = params[str(run)]['mean_60']
        sig_60 = params[str(run)]['sig_60']
        chiSquare_fit_60 = params[str(run)]['chiSquare_fit_60']
        # cut_60_3sig = params[str(run)]['cut_60_3sig']
        # bkg_60 = params[str(run)]['bkg_60']
        
        cut_60_3sig = params[str(run)]['cut_60_3sig']
        bkg_60 = params[str(run)]['bkg_60']
        
        counts_60_3sig = params[str(run)]['counts_peak'] # total counts within 3 sigma of 60 keV fit mean 
                                                         # (incl. background)
        counts_bkg = params[str(run)]['counts_bkg'] # total counts in background region left and right of 60keV
                                                    # peak (1.5 sigma left and 1.5 sigma right)
        
        # counts_60_3sig = len(df.query(cut_60_3sig).copy()) # total counts within 3 sigma of 60 keV fit mean 
                                                           # (incl. background)
        # counts_bkg = len(df.query(bkg_60).copy()) # total counts in background region left and right of 60keV peak 
                                                  # (1.5 sigma left and 1.5 sigma right)
            
        counts_peak = counts_60_3sig - counts_bkg # background subtracted counts in 60 keV peak region
        
        # get waveforms from the 60 keV and sideband regions
        times, all_60_raw = cage_utils.get_superpulse_taligned(df, dg, cut_60_3sig, all=True, norm=False)
        times, bkg_60_raw = cage_utils.get_superpulse_taligned(df, dg, bkg_60, all=True, norm=False)
        
        # construct "pure" 60 keV gamma superpulse from 60 keV - sideband, weighted by counts in peak vs sideband
        bkg_sub_60_raw = (counts_60_3sig/counts_peak)*(all_60_raw - (counts_bkg/counts_60_3sig)*bkg_60_raw)
        
        all_60_raw_notched = cage_utils.notchFilter_SIS3302(all_60_raw, Q=20)
        bkg_60_raw_notched = cage_utils.notchFilter_SIS3302(bkg_60_raw, Q=20)
        bkg_sub_60_raw_notched = cage_utils.notchFilter_SIS3302(bkg_sub_60_raw, Q=20)
        
        all_60 = np.divide(all_60_raw_notched, np.amax(all_60_raw_notched))
        bkg_60 = np.divide(bkg_60_raw_notched, np.amax(bkg_60_raw_notched))
        bkg_sub_60 = np.divide(bkg_sub_60_raw_notched, np.amax(bkg_sub_60_raw_notched))
        
        all_60_raw_pz = cage_utils.double_pole_zero([all_60_raw_notched], 21250, 433, 0.045)[0]
        bkg_60_raw_pz = cage_utils.double_pole_zero([bkg_60_raw_notched], 21250, 433, 0.045)[0]
        bkg_sub_60_raw_pz = cage_utils.double_pole_zero([bkg_sub_60_raw_notched], 21250, 433, 0.045)[0]
        
        Etrap_all_60 = cage_utils.trap_norm(all_60_raw_pz, 100, 400)
        Etrap_bkg_60 = cage_utils.trap_norm(bkg_60_raw_pz, 100, 400)
        Etrap_bkg_sub_60 = cage_utils.trap_norm(bkg_sub_60_raw_pz, 100, 400)
        
        trap0_all_60 = cage_utils.asymTrapFilter(all_60_raw_pz, 100, 1, 400)
        trap0_bkg_60 = cage_utils.asymTrapFilter(bkg_60_raw_pz, 100, 1, 400)
        trap0_bkg_sub_60 = cage_utils.asymTrapFilter(bkg_sub_60_raw_pz, 100, 1, 400)
        
        max_trap0_all_60 = np.argmax(trap0_all_60)
        max_trap0_bkg_60 = np.argmax(trap0_bkg_60)
        max_trap0_bkg_sub_60 = np.argmax(trap0_bkg_sub_60)
        
        t0_all_60 = cage_utils.time_point_thresh_max(trap0_all_60, 0.0, max_trap0_all_60, 
                                                     max_trap0_all_60 - 200)
        t0_bkg_60 = cage_utils.time_point_thresh_max(trap0_bkg_60, 0.0, max_trap0_bkg_60, 
                                                     max_trap0_bkg_60 - 200)
        t0_bkg_sub_60 = cage_utils.time_point_thresh_max(trap0_bkg_sub_60, 0.0, max_trap0_bkg_sub_60, 
                                                          max_trap0_bkg_sub_60 - 200)
        print(f't0: {t0_bkg_sub_60}')

        Eftp_all_60 = Etrap_all_60[t0_all_60 + 400]
        Eftp_bkg_60 = Etrap_bkg_60[t0_bkg_60 + 400]
        Eftp_bkg_sub_60 = Etrap_bkg_sub_60[t0_bkg_sub_60 + 400]
        
        print(f'Eftp: {Eftp_bkg_sub_60}')
        
        all_60_pz = np.divide(all_60_raw_pz, Eftp_all_60)
        bkg_60_pz = np.divide(bkg_60_raw_pz, Eftp_bkg_60)
        bkg_sub_60_pz = np.divide(bkg_sub_60_raw_pz, Eftp_bkg_sub_60)

        max_all_60 = np.argmax(all_60_raw_notched)
        max_bkg_60 = np.argmax(bkg_60_raw_notched)
        max_bkg_sub_60 = np.argmax(bkg_sub_60_raw_notched)
        
        all_60_pz_max = np.divide(all_60_raw_pz, all_60_raw_pz[max_all_60])
        bkg_60_pz_max = np.divide(bkg_60_raw_pz, bkg_60_raw_pz[max_bkg_60])
        bkg_sub_60_pz_max = np.divide(bkg_sub_60_raw_pz, bkg_sub_60_raw_pz[max_bkg_sub_60])
        
        if idx ==0:
            print('Creating initial dataframe')
            wf_dict = {'run': run, 
                       'radius': radius, 
                       'counts_bkg': counts_bkg, 
                       'counts_60_3sig': counts_60_3sig, 
                       'counts_pure_60': counts_peak, 
                       'samples': [times], 
                       'bkg_and_60': [all_60], 
                       'bkg': [bkg_60], 
                       'pure_60': [bkg_sub_60], 
                       'bkg_and_60_raw': [all_60_raw], 
                       'bkg_raw': [bkg_60_raw], 
                       'pure_60_raw': [bkg_sub_60_raw], 
                       'bkg_and_60_raw_pz': [all_60_raw_pz], 
                       'bkg_raw_pz': [bkg_60_raw_pz], 
                       'pure_60_raw_pz': [bkg_sub_60_raw_pz], 
                       'bkg_and_60_pz': [all_60_pz], 
                       'bkg_pz': [bkg_60_pz], 
                       'pure_60_pz': [bkg_sub_60_pz], 
                       'bkg_and_60_pz_max': [all_60_pz_max], 
                       'bkg_pz_max': [bkg_60_pz_max], 
                       'pure_60_pz_max': [bkg_sub_60_pz],
                       'Etrap_all_60': [Etrap_all_60], 
                       'Etrap_bkg_60': [Etrap_bkg_60],
                       'Etrap_bkg_sub_60': [Etrap_bkg_sub_60],
                       'trap0_all_60': [trap0_all_60],
                       'trap0_bkg_60': [trap0_bkg_60],
                       'trap0_bkg_sub_60': [trap0_bkg_sub_60],
                       'argmax_bkg_and_60': max_all_60, 
                       'argmax_bkg': max_bkg_60, 
                       'argmax_pure_60': max_bkg_sub_60, 
                       'Eftp_all_60': Eftp_all_60, 
                       'Eftp_bkg_60': Eftp_bkg_60, 
                       'Eftp_bkg_sub_60': Eftp_bkg_sub_60,
                       't0_all_60': t0_all_60, 
                       't0_bkg_60': t0_bkg_60, 
                       't0_bkg_sub_60': t0_bkg_sub_60}
            
            wf_df = pd.DataFrame(data=wf_dict)
            
        else:
            print('appending to dataframe')
            wf_dict = {'run': run, 
                       'radius': radius, 
                       'counts_bkg': counts_bkg, 
                       'counts_60_3sig': counts_60_3sig, 
                       'counts_pure_60': counts_peak, 
                       'samples': [times], 
                       'bkg_and_60': [all_60], 
                       'bkg': [bkg_60], 
                       'pure_60': [bkg_sub_60], 
                       'bkg_and_60_raw': [all_60_raw], 
                       'bkg_raw': [bkg_60_raw], 
                       'pure_60_raw': [bkg_sub_60_raw], 
                       'bkg_and_60_raw_pz': [all_60_raw_pz], 
                       'bkg_raw_pz': [bkg_60_raw_pz], 
                       'pure_60_raw_pz': [bkg_sub_60_raw_pz], 
                       'bkg_and_60_pz': [all_60_pz], 
                       'bkg_pz': [bkg_60_pz], 
                       'pure_60_pz': [bkg_sub_60_pz], 
                       'bkg_and_60_pz_max': [all_60_pz_max], 
                       'bkg_pz_max': [bkg_60_pz_max], 
                       'pure_60_pz_max': [bkg_sub_60_pz],
                       'Etrap_all_60': [Etrap_all_60], 
                       'Etrap_bkg_60': [Etrap_bkg_60],
                       'Etrap_bkg_sub_60': [Etrap_bkg_sub_60],
                       'trap0_all_60': [trap0_all_60],
                       'trap0_bkg_60': [trap0_bkg_60],
                       'trap0_bkg_sub_60': [trap0_bkg_sub_60],
                       'argmax_bkg_and_60': max_all_60, 
                       'argmax_bkg': max_bkg_60, 
                       'argmax_pure_60': max_bkg_sub_60, 
                       'Eftp_all_60': Eftp_all_60, 
                       'Eftp_bkg_60': Eftp_bkg_60, 
                       'Eftp_bkg_sub_60': Eftp_bkg_sub_60,
                       't0_all_60': t0_all_60, 
                       't0_bkg_60': t0_bkg_60, 
                       't0_bkg_sub_60': t0_bkg_sub_60}
            
            wf_df_temp = pd.DataFrame(data=wf_dict)
            
            wf_df = wf_df.append(wf_df_temp, ignore_index=True)
            
    if write_data==True:
        outfile = f'./data/normScan/superpulses_60keV_allRuns.hdf5'
        # outfile = f'./data/normScan/new_superpulses_60keV_allRuns.hdf5'
        # outfile = f'./data/normScan/superpulses_2sig_60keV_allRuns.hdf5'
        # outfile = f'./data/normScan/superpulses_1sig_60keV_allRuns.hdf5'
        wf_df.to_hdf(outfile, key='superpulses', mode='w')
        
def risetime_dists(run_list, campaign, dsp_list, norm = False, user=True, hit=True, cal=True, lowE=False,
                   savefig=False, writeParams=False):
    

    for run in run_list:
        df_raw, dg, runtype, rt_min, radius, angle_det, rotary = cage_utils.getDataFrame(run, user=user, hit=hit, cal=cal, dsp_list=dsp_list, lowE=lowE)
    
        df = cage_utils.apply_DC_Cuts(run, df_raw)
        
        with open('./analysis_60keV.json') as f:
            params = json.load(f)

        cut_60_3sig = params[str(run)]['cut_60_3sig']
        bkg_60 = params[str(run)]['bkg_60']
        
        df_60 = df.query(cut_60_3sig).copy()
        df_bkg = df.query(bkg_60).copy()
        
        
        # tp_list = ['tp0020', 'tp0030', 'tp0050', 'tp0220', 'tp0250', 'tp0530', 'tp0550', 'tp1090']
        tp_list = ['tp0020', 'tp0050', 'tp0220', 'tp0250', 'tp0530', 'tp0550', 'tp1090']
        

        df_60['tp0020'] = df_60['tp_20'] - df_60['tp_0']
        df_60['tp0030'] = df_60['tp_30'] - df_60['tp_0']
        df_60['tp0050'] = df_60['tp_50'] - df_60['tp_0']
        df_60['tp0220'] = df_60['tp_20'] - df_60['tp_02']
        df_60['tp0250'] = df_60['tp_50'] - df_60['tp_02']
        df_60['tp0530'] = df_60['tp_30'] - df_60['tp_05']
        df_60['tp0550'] = df_60['tp_50'] - df_60['tp_05']
        
        df_60['tp1090'] = df_60['tp_90'] - df_60['tp_10']
        df_60['tp10max'] = df_60['tp_max'] - df_60['tp_10']

        df_bkg['tp0020'] = df_bkg['tp_20'] - df_bkg['tp_0']
        df_bkg['tp0030'] = df_bkg['tp_30'] - df_bkg['tp_0']
        df_bkg['tp0050'] = df_bkg['tp_50'] - df_bkg['tp_0']
        df_bkg['tp0210'] = df_bkg['tp_10'] - df_bkg['tp_02']
        df_bkg['tp0220'] = df_bkg['tp_20'] - df_bkg['tp_02']
        df_bkg['tp0250'] = df_bkg['tp_50'] - df_bkg['tp_02']
        df_bkg['tp0530'] = df_bkg['tp_30'] - df_bkg['tp_05']
        df_bkg['tp0550'] = df_bkg['tp_50'] - df_bkg['tp_05']
        
        df_bkg['tp1090'] = df_bkg['tp_90'] - df_bkg['tp_10']
        df_bkg['tp10max'] = df_bkg['tp_max'] - df_bkg['tp_10']
        
        for tp in tp_list:
            print(tp)
            
            # first get get tp in signal region
            
            fig, ax = plt.subplots()
            tlo, thi, tpb = 0, 800, 10
            nbx = int((thi-tlo)/tpb)


            tp_hist_60, bins = np.histogram(df_60[tp], bins=nbx,
                                                range=[tlo, thi])

            plt.plot(bins[1:], tp_hist_60, ds='steps', c='b', lw=1)

            plt.xlabel(f'{tp} (ns)', fontsize=16)
            plt.ylabel('counts', fontsize=16)

            plt.title(f'60 keV {tp} \nrun {run}, {radius} mm', fontsize = 16)

            plt.setp(ax.get_xticklabels(), fontsize=14)
            plt.setp(ax.get_yticklabels(), fontsize=14)
            
            plt.tight_layout()
            
            if savefig==True:
                plt.savefig(f'./plots/{campaign}risetimes/new/run{run}_60keV_{tp}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}risetimes/new/run{run}_60keV_{tp}.pdf', dpi=200)

            plt.clf()
            plt.close()
            
            fig, ax = plt.subplots()
            tlo, thi, tpb = 0, 800, 10
            nbx = int((thi-tlo)/tpb)
            
            # get tp in background/sideband region


            tp_hist_bkg, bins = np.histogram(df_bkg[tp], bins=nbx,
                                                range=[tlo, thi])

            plt.plot(bins[1:], tp_hist_bkg, ds='steps', c='b', lw=1)

            plt.xlabel(f'{tp} (ns)', fontsize=16)
            plt.ylabel('counts/10 ns', fontsize=16)

            plt.title(f'Background {tp} \nrun {run}, {radius} mm', fontsize = 16)

            plt.setp(ax.get_xticklabels(), fontsize=14)
            plt.setp(ax.get_yticklabels(), fontsize=14)
            
            plt.tight_layout()
            
            if savefig==True:
                plt.savefig(f'./plots/{campaign}risetimes/new/run{run}_bkg_{tp}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}risetimes/new/run{run}_bkg_{tp}.pdf', dpi=200)

            plt.clf()
            plt.close()
            
            # fit the background tp distribution
            
            if (tp=='tp0020') or (tp=='tp0030') or (tp=='tp0050'):
                bkg_pars, bkg_cov = pgf.gauss_mode_width_max(tp_hist_bkg, bins, n_bins = 5)
            else:
                bkg_pars, bkg_cov = pgf.gauss_mode_width_max(tp_hist_bkg, bins, n_bins = 10)

            bkg_mode = bkg_pars[0]
            bkg_width = bkg_pars[1]
            bkg_amp = bkg_pars[2]
            
            bkg_fit_pars, bkg_fit_cov = pgf.fit_hist(pgf.gauss_basic, tp_hist_bkg, bins, guess = (bkg_mode, 
                                                                                bkg_width, bkg_amp, 0))
            
            bkg_mean = bkg_fit_pars[0]
            bkg_sig = bkg_fit_pars[1]
            
            # print(f'Mean guess 1: {bkg_mean} \nSigma guess 1: {bkg_sig}')
            
            # second fit of only subsection around 5 sigma of original peak for better fit
            
            fit_lo, fit_hi = bkg_mean - 5*bkg_sig, bkg_mean + 5*bkg_sig
            
            if fit_lo <0:
                fit_lo = 0.
            nbx_fit = int((fit_hi - fit_lo)/10)
            
            tp_hist_bkg_fit, bins_fit = np.histogram(df_bkg[tp], bins=nbx_fit,
                                                range=[fit_lo, fit_hi])
            bkg_fit2_pars, bkg_fit2_cov = pgf.fit_hist(pgf.gauss_basic, tp_hist_bkg_fit, bins_fit, 
                                                       guess = (bkg_mean, bkg_sig, bkg_amp, 0))
            
            bkg_mean_fit = bkg_fit2_pars[0]
            bkg_sig_fit = bkg_fit2_pars[1]
            
            bkg_chi_2 = pgf.goodness_of_fit(tp_hist_bkg_fit, bins_fit, pgf.gauss_basic, bkg_fit2_pars)
            
            fig, ax = plt.subplots()
            
            
            plt.plot(bins_fit[1:], pgf.gauss_basic(bins_fit[1:], *bkg_fit2_pars), c = 'r')
            plt.plot(bins[1:], tp_hist_bkg, ds='steps', c='b', lw=1)
            
            ax.text(0.95, 0.8,  f'mean: {bkg_mean_fit:.0f} ns \nsigma: {bkg_sig_fit:.0f} ns', 
                verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes, color='black',
                fontsize=16, bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 8})
            
            plt.xlabel(f'{tp} (ns)', fontsize=16)
            plt.ylabel('counts/ 10 ns', fontsize=16)

            plt.title(f'Background {tp} with fit \nrun {run}, {radius} mm', fontsize = 16)

            plt.setp(ax.get_xticklabels(), fontsize=14)
            plt.setp(ax.get_yticklabels(), fontsize=14)
            
            plt.tight_layout()
            
            if savefig==True:
                plt.savefig(f'./plots/{campaign}risetimes/new/run{run}_bkg_fit_{tp}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}risetimes/new/run{run}_bkg_fit_{tp}.pdf', dpi=200)

            plt.clf()
            plt.close()

            bkg_cut_3sig_lo = bkg_mean_fit-3*bkg_sig_fit
            bkg_cut_3sig_hi = bkg_mean_fit+3*bkg_sig_fit
            
            bkg_counts = len(df_bkg.query(f'({tp} > {bkg_cut_3sig_lo}) and ({tp} < {bkg_cut_3sig_hi})'))
            
            # _____________________________________________
            # now do background subtraction of signal hist and sideband hist
            
            fig, ax = plt.subplots()
            tlo, thi, tpb = 0, 800, 10
            nbx = int((thi-tlo)/tpb)
            
            tp_hist_bkg_sub = tp_hist_60 - tp_hist_bkg
            tp_hist_bkg_sub = tp_hist_bkg_sub.clip(min=0.0)
            plt.plot(bins[1:], tp_hist_bkg_sub, ds='steps', c='b', lw=1)


            plt.xlabel(f'{tp} (ns)', fontsize=16)
            plt.ylabel('counts', fontsize=16)

            plt.title(f'Background-subtracted {tp} \nrun {run}, {radius} mm', fontsize = 16)

            plt.setp(ax.get_xticklabels(), fontsize=14)
            plt.setp(ax.get_yticklabels(), fontsize=14)
            
            plt.tight_layout()
            
            if savefig==True:
                plt.savefig(f'./plots/{campaign}risetimes/new/run{run}_bkg_sub_{tp}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}risetimes/new/run{run}_bkg_sub_{tp}.pdf', dpi=200)
            
            plt.clf()
            plt.close()
            
            # fit background-subtracted hist
            
            pars, cov = pgf.gauss_mode_width_max(tp_hist_bkg_sub, bins, n_bins = 10)
            mode = pars[0]
            width = pars[1]
            amp = pars[2]
            
            fit_pars, fit_cov = pgf.fit_hist(pgf.gauss_basic, tp_hist_bkg_sub, bins, 
                                             guess = (mode, width, amp, 0))
            
            mean = fit_pars[0]
            sig = fit_pars[1]
            
            # print(f'Mean guess: {mean} \nSigma guess: {sig}')
            
            #second fit of only the peak region. First get a subsection of the bkg-sub hist only near the peak
            
            fit_lo, fit_hi = mean - 5*sig, mean + 5*sig
            
            # get bin numbers corresponding to +/- 5 sigma of the peak 
            fit_lims = [fit_lo, fit_hi]
            idxs = np.digitize(fit_lims, bins)
            
            idx_lo = idxs[0]
            idx_hi = idxs[1]
            
            bins_new = bins[idx_lo:idx_hi]
            y = tp_hist_bkg_sub[idx_lo:idx_hi-1]
                
            # print(f'len bins: {len(bins_new)} \nlength y: {len(y)}')
            
            # print(bins_new)
            # print(y)
            
            # just a test to see if new region being chosen correctly
            # plt.plot(bins_new[1:], y, ds='steps', c='b', lw=1)
            # plt.savefig(f'./plots/{campaign}risetimes/new/run{run}_test_subHist_{tp}.png', dpi=200)
            # plt.clf()
            # plt.close()
            
            # perform second fit of region around the peak only

            fit2_pars, fit2_cov = pgf.fit_hist(pgf.gauss_basic, y, bins_new, 
                                                       guess = (mean, sig, amp, 0))
            
            mean_final = fit2_pars[0]
            sig_final = fit2_pars[1]
            
            chi_2 = pgf.goodness_of_fit(tp_hist_bkg_sub, bins, pgf.gauss_basic, fit2_pars)
            
            fig, ax = plt.subplots()
            
            # plot fit from small region around the peak over entire tp range 
            plt.plot(bins_new[1:], pgf.gauss_basic(bins_new[1:], *fit2_pars), c = 'r')
            plt.plot(bins[1:], tp_hist_bkg_sub, ds='steps', c='b', lw=1)
            # exit()
            
            ax.text(0.95, 0.8,  f'mean: {mean_final:.0f} ns \nsigma: {sig_final:.0f} ns', 
                verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes, color='black',
                fontsize=16, bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 8})
            
            plt.xlabel(f'{tp} (ns)', fontsize=16)
            plt.ylabel('counts/10 ns', fontsize=16)

            plt.title(f'Background-subtracted {tp} with fit \nrun {run}, {radius} mm', fontsize = 16)

            plt.setp(ax.get_xticklabels(), fontsize=14)
            plt.setp(ax.get_yticklabels(), fontsize=14)
            
            plt.tight_layout()
            
            if savefig==True:
                plt.savefig(f'./plots/{campaign}risetimes/new/run{run}_bkg_sub_fit_{tp}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}risetimes/new/run{run}_bkg_sub_fit_{tp}.pdf', dpi=200)

            plt.clf()
            plt.close()
            
            # count the number of events in the background-subtracted peak
            
            tp_hist_bkg_sub_clip = tp_hist_bkg_sub.clip(min=0.0)

            
            cut_3sig_lo = mean_final-3*sig_final
            cut_3sig_hi = mean_final+3*sig_final
            
            # get bin numbers corresponding to these values
            cuts = [cut_3sig_lo, cut_3sig_hi]
            idxs = np.digitize(cuts, bins)

            print(cut_3sig_lo, cut_3sig_hi)
            
            idx_lo = idxs[0]
            idx_hi = idxs[1]

            # print(idx_lo, idx_hi)
            
            # sum the part of the histogram corresponding to those indices
            # the way it works, getting [idx_lo(hi) - 1] is the actual bin position of cut_3sig_lo()hi
            # but have to add one to it to get the histogram to include cut_3sig_hi, which is why the 
            # range is [idx_lo-1:idx_hi]
            counts = np.sum(tp_hist_bkg_sub_clip[idx_lo-1:idx_hi])

            print(f'counts: {counts}')
            
            
            if writeParams==True:
                cage_utils.writeJson('./analysis_60keV.json', run, f'mean_{tp}', mean)
                cage_utils.writeJson('./analysis_60keV.json', run, f'sig_{tp}', sig)
                cage_utils.writeJson('./analysis_60keV.json', run, f'counts_3sig_{tp}', counts)
                cage_utils.writeJson('./analysis_60keV.json', run, f'bkg_mean_{tp}', bkg_mean)
                cage_utils.writeJson('./analysis_60keV.json', run, f'bkg_sig_{tp}', bkg_sig)
                cage_utils.writeJson('./analysis_60keV.json', run, f'bkg_counts_3sig_{tp}', bkg_counts)
                
def risetime_superpulses(run_list, campaign, dsp_list, norm = False, user=True, hit=True, cal=True,
                         lowE=False, savefig=True):
    
    # f_superpulse = './data/normScan/superpulses_60keV_allRuns.hdf5'
    # f_superpulse = './data/normScan/new_superpulses_60keV_allRuns.hdf5'
    f_superpulse = f'./data/normScan/superpulses_2sig_60keV_allRuns.hdf5'
    # f_superpulse = f'./data/normScan/superpulses_1sig_60keV_allRuns.hdf5'
    data_superpulse = pd.read_hdf(f_superpulse, key = '/superpulses')
    
    campaign+='2sig/'
    
    # first calculate timepoints from superpulses
    
    wf_arr = np.array(data_superpulse['pure_60_pz'])
    bkg_wf_arr = np.array(data_superpulse['bkg_pz'])
    bkg_wf = bkg_wf_arr[0]
    times = np.array(data_superpulse['samples'][0])
    
    rad_arr = np.array(data_superpulse['radius'])
    arg_maxes = np.array(data_superpulse['argmax_pure_60'])
    bkg_arg_maxes = np.array(data_superpulse['argmax_bkg'])
    
    Eftp = np.array(data_superpulse['Eftp_bkg_sub_60'])
    bkg_Eftp = np.array(data_superpulse['Eftp_bkg_60'])

    t0 = np.array(data_superpulse['t0_bkg_sub_60'])
    t0_bkg = np.array(data_superpulse['t0_bkg_60'])
    
    rt_0220 = np.zeros(len(wf_arr))
    rt_0250 = np.zeros(len(wf_arr))
    rt_0530 = np.zeros(len(wf_arr))
    rt_0550 = np.zeros(len(wf_arr))
    rt_1090 = np.zeros(len(wf_arr))
    rt_0020 = np.zeros(len(wf_arr))
    rt_0030 = np.zeros(len(wf_arr))
    rt_0050 = np.zeros(len(wf_arr))
    
    # using max of WF overshoot
    # for n in range(len(wf_arr)):
        # t_02 = cage_utils.time_point_frac(wf_arr[n], 0.02, arg_maxes[n])
        # t_05 = cage_utils.time_point_frac(wf_arr[n], 0.05, arg_maxes[n])
        # t_10 = cage_utils.time_point_frac(wf_arr[n], 0.1, arg_maxes[n])
        # t_20 = cage_utils.time_point_frac(wf_arr[n], 0.2, arg_maxes[n])
        # t_30 = cage_utils.time_point_frac(wf_arr[n], 0.3, arg_maxes[n])
        # t_50 = cage_utils.time_point_frac(wf_arr[n], 0.5, arg_maxes[n])
        # t_90 = cage_utils.time_point_frac(wf_arr[n], 0.9, arg_maxes[n])
        # rt_1090[n] = (t_90-t_10)*10 #in ns
        # rt_0550[n] = (t_50-t_05)*10 #in ns
        # rt_0530[n] = (t_30-t_05)*10 #in ns
        # rt_0250[n] = (t_50-t_02)*10 #in ns
        # rt_0220[n] = (t_20-t_02)*10 #in ns
        # rt_0020[n] = (t_20-t0[n])*10 #in ns
        # rt_0030[n] = (t_30-t0[n])*10 #in ns
        # rt_0050[n] = (t_50-t0[n])*10 #in ns
        
    # using Eftp from superpulse    
    for n in range(len(wf_arr)):
        t_02 = cage_utils.time_point_frac(wf_arr[n], 0.02, t0[n] + 400)
        t_05 = cage_utils.time_point_frac(wf_arr[n], 0.05, t0[n] + 400)
        t_10 = cage_utils.time_point_frac(wf_arr[n], 0.1, t0[n] + 400)
        t_20 = cage_utils.time_point_frac(wf_arr[n], 0.2, t0[n] + 400)
        t_30 = cage_utils.time_point_frac(wf_arr[n], 0.3, t0[n] + 400)
        t_50 = cage_utils.time_point_frac(wf_arr[n], 0.5, t0[n] + 400)
        t_90 = cage_utils.time_point_frac(wf_arr[n], 0.9, t0[n] + 400)
        rt_1090[n] = (t_90-t_10)*10 #in ns
        rt_0550[n] = (t_50-t_05)*10 #in ns
        rt_0530[n] = (t_30-t_05)*10 #in ns
        rt_0250[n] = (t_50-t_02)*10 #in ns
        rt_0220[n] = (t_20-t_02)*10 #in ns
        rt_0020[n] = (t_20-t0[n])*10 #in ns
        rt_0030[n] = (t_30-t0[n])*10 #in ns
        rt_0050[n] = (t_50-t0[n])*10 #in ns
        
    # background values
    bkg_rt_0220 = np.zeros(len(bkg_wf_arr))
    bkg_rt_0250 = np.zeros(len(bkg_wf_arr))
    bkg_rt_0530 = np.zeros(len(bkg_wf_arr))
    bkg_rt_0550 = np.zeros(len(bkg_wf_arr))
    bkg_rt_1090 = np.zeros(len(bkg_wf_arr))
    bkg_rt_0020 = np.zeros(len(bkg_wf_arr))
    bkg_rt_0030 = np.zeros(len(bkg_wf_arr))
    bkg_rt_0050 = np.zeros(len(bkg_wf_arr))
    
    # using max of WF overshoot
    # for n in range(len(bkg_wf_arr)):
        # bkg_t_02 = cage_utils.time_point_frac(bkg_wf_arr[n], 0.02, bkg_arg_maxes[n])
        # bkg_t_05 = cage_utils.time_point_frac(bkg_wf_arr[n], 0.05, bkg_arg_maxes[n])
        # bkg_t_10 = cage_utils.time_point_frac(bkg_wf_arr[n], 0.1, bkg_arg_maxes[n])
        # bkg_t_20 = cage_utils.time_point_frac(bkg_wf_arr[n], 0.2, bkg_arg_maxes[n])
        # bkg_t_30 = cage_utils.time_point_frac(bkg_wf_arr[n], 0.3, bkg_arg_maxes[n])
        # bkg_t_50 = cage_utils.time_point_frac(bkg_wf_arr[n], 0.5, bkg_arg_maxes[n])
        # bkg_t_90 = cage_utils.time_point_frac(bkg_wf_arr[n], 0.9, bkg_arg_maxes[n])
        # bkg_rt_1090[n] = (bkg_t_90-bkg_t_10)*10 #in ns
        # bkg_rt_0550[n] = (bkg_t_50-bkg_t_05)*10 #in ns
        # bkg_rt_0530[n] = (bkg_t_30-bkg_t_05)*10 #in ns
        # bkg_rt_0250[n] = (bkg_t_50-bkg_t_02)*10 #in ns
        # bkg_rt_0220[n] = (bkg_t_20-bkg_t_02)*10 #in ns
        # bkg_rt_0020[n] = (bkg_t_20-bkg_t0[n])*10 #in ns
        # bkg_rt_0030[n] = (bkg_t_30-bkg_t0[n])*10 #in ns
        # bkg_rt_0050[n] = (bkg_t_50-bkg_t0[n])*10 #in ns
        
    # using Eftp from superpulse
    for n in range(len(bkg_wf_arr)):
        bkg_t_02 = cage_utils.time_point_frac(bkg_wf_arr[n], 0.02, t0_bkg[n] + 400)
        bkg_t_05 = cage_utils.time_point_frac(bkg_wf_arr[n], 0.05, t0_bkg[n] + 400)
        bkg_t_10 = cage_utils.time_point_frac(bkg_wf_arr[n], 0.1, t0_bkg[n] + 400)
        bkg_t_20 = cage_utils.time_point_frac(bkg_wf_arr[n], 0.2, t0_bkg[n] + 400)
        bkg_t_30 = cage_utils.time_point_frac(bkg_wf_arr[n], 0.3, t0_bkg[n] + 400)
        bkg_t_50 = cage_utils.time_point_frac(bkg_wf_arr[n], 0.5, t0_bkg[n] + 400)
        bkg_t_90 = cage_utils.time_point_frac(bkg_wf_arr[n], 0.9, t0_bkg[n] + 400)
        bkg_rt_1090[n] = (bkg_t_90-bkg_t_10)*10 #in ns
        bkg_rt_0550[n] = (bkg_t_50-bkg_t_05)*10 #in ns
        bkg_rt_0530[n] = (bkg_t_30-bkg_t_05)*10 #in ns
        bkg_rt_0250[n] = (bkg_t_50-bkg_t_02)*10 #in ns
        bkg_rt_0220[n] = (bkg_t_20-bkg_t_02)*10 #in ns
        bkg_rt_0020[n] = (bkg_t_20-t0_bkg[n])*10 #in ns
        bkg_rt_0030[n] = (bkg_t_30-t0_bkg[n])*10 #in ns
        bkg_rt_0050[n] = (bkg_t_50-t0_bkg[n])*10 #in ns
        
    tp_list = ['tp0020', 'tp0050', 'tp0220', 'tp0250', 'tp0530', 'tp0550', 'tp1090']
    tp_values = np.array([rt_0020, rt_0050, rt_0220, rt_0250, rt_0530, rt_0550, rt_1090])
    # tp0_list = ['tp0020', 'tp0030', 'tp0050']
    # tp0_values = np.array([rt_0020, rt_0030, rt_0050])
    # tp0_err = np.repeat(10., len(rad_arr))
    bkg_tp_values = np.array([bkg_rt_0020, bkg_rt_0050, bkg_rt_0220, bkg_rt_0250, bkg_rt_0530, 
                              bkg_rt_0550, bkg_rt_1090])
    
    
    # get uncertainties from fits of distribution obtained from risetime_dists()
    
    with open('./analysis_60keV.json') as f:
        params = json.load(f)
    
    tp_full_arr = np.zeros(shape = (len(tp_list), len(rad_arr)))
    tp_full_err_arr = np.zeros(shape = (len(tp_list), len(rad_arr)))
    bkg_tp_full_arr = np.zeros(shape = (len(tp_list), len(rad_arr)))
    bkg_tp_full_err_arr = np.zeros(shape = (len(tp_list), len(rad_arr)))
    for (tp, n) in zip(tp_list, range(len(tp_list))):
        tp_arr = np.zeros(len(rad_arr))
        tp_err_arr = np.zeros(len(rad_arr))
        bkg_tp_arr = np.zeros(len(rad_arr))
        bkg_tp_err_arr = np.zeros(len(rad_arr))
        for (run, m) in zip(run_list, range(len(rad_arr))):
            print(f'run: {run} \ntp: {tp}')
            tp_mean = params[str(run)][f'mean_{tp}']
            tp_sig = params[str(run)][f'sig_{tp}']
            tp_counts = params[str(run)][f'counts_3sig_{tp}']
            tp_err = np.sqrt(tp_sig**2/tp_counts+(10**2)) # add err from gaussian fit in quadrature with 
                                                          # precision of digitizer (10 ns)
            print(f'tp_counts: {tp_counts}\ntp_err: {tp_err}')
            tp_arr[m] = tp_mean
            tp_err_arr[m] = tp_err
            
            bkg_tp_mean = params[str(run)][f'bkg_mean_{tp}']
            bkg_tp_sig = params[str(run)][f'bkg_sig_{tp}']
            bkg_tp_counts = params[str(run)][f'bkg_counts_3sig_{tp}']
            bkg_tp_err = (bkg_tp_sig/np.sqrt(bkg_tp_counts))+10
            print(f'bkg_tp_counts: {bkg_tp_counts}\nbkg_tp_err: {bkg_tp_err}')
            # tp_err = tp_sig
            bkg_tp_arr[m] = bkg_tp_mean
            bkg_tp_err_arr[m] = bkg_tp_err
        tp_full_arr[n] = tp_arr
        tp_full_err_arr[n] = tp_err_arr
        bkg_tp_full_arr[n] = bkg_tp_arr
        bkg_tp_full_err_arr[n] = bkg_tp_err_arr
        
        fig, ax = plt.subplots()
        
        plt.errorbar(rad_arr, tp_full_arr[n], yerr=tp_full_err_arr[n], marker = '.', c='g', ls='none',
                     label=f'distribution')
        plt.errorbar(rad_arr, tp_values[n], yerr=tp_full_err_arr[n], marker = '.', c='b', ls='none',
                     label=f'60 keV superpulse')
        plt.errorbar(rad_arr, bkg_tp_values[n], yerr=bkg_tp_full_err_arr[n], marker = '.', c='r', ls='none',
                     label=f'bkg superpulse')

    
        plt.xlabel('Radial Position (mm)', fontsize=14)
        plt.ylabel(f'{tp} risetime (ns)', fontsize=14)
        # plt.title(f'{tp} Risetime VS Radius', fontsize=14)
        plt.title(f'{tp} Risetime VS Radius', fontsize=14)
        plt.setp(ax.get_xticklabels(), fontsize=12)
        plt.setp(ax.get_yticklabels(), fontsize=12)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        if savefig:
            plt.savefig(f'./plots/{campaign}risetimes/{tp}_vs_rad.png', dpi=200)
            plt.savefig(f'./plots/{campaign}risetimes/{tp}_vs_rad.pdf', dpi=200)
            # plt.savefig(f'./plots/{campaign}risetimes/new_{tp}_vs_rad.png', dpi=200)
    
        plt.clf()
        plt.close()
        
    # plot all risetimes on same plot
    markers = ['o', '^', 'X', 'v', 'P', 'D', 's', '>']
    col = ['r', 'b', 'g', 'c', 'm', 'k', 'r', 'b']
    fig, ax = plt.subplots()
    for (tp, n) in zip(tp_list, range(len(tp_list))):
        plt.errorbar(rad_arr, tp_full_arr[n], yerr=tp_full_err_arr[n], marker = markers[n], c=col[n], 
                     ls='none', label=f'{tp}')
    plt.xlabel('Radial Position (mm)', fontsize=14)
    plt.ylabel(f'risetime (ns)', fontsize=14)
    # plt.title(f'Risetime VS Radius \nfrom distribution', fontsize=14)
    plt.title(f'Risetime VS Radius for 60 keV \nfrom distribution', fontsize=14)
    plt.setp(ax.get_xticklabels(), fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)
    plt.legend(fontsize=12, bbox_to_anchor=(1.04,1), borderaxespad=0)
    plt.tight_layout()
        
    if savefig:
        plt.savefig(f'./plots/{campaign}risetimes/rt_all_dist_vs_rad.png', dpi=200)
        plt.savefig(f'./plots/{campaign}risetimes/rt_all_dist_vs_rad.pdf', dpi=200)
        # plt.savefig(f'./plots/{campaign}risetimes/new_rt_all_dist_vs_rad.png', dpi=200)
    
    plt.clf()
    plt.close()
    
    # do same for risetimes from distributions in sideband regions
    fig, ax = plt.subplots()
    for (tp, n) in zip(tp_list, range(len(tp_list))):
        plt.errorbar(rad_arr, bkg_tp_full_arr[n], yerr=bkg_tp_full_err_arr[n], marker = markers[n], c=col[n], 
                     ls='none', label=f'{tp}')
    plt.xlabel('Radial Position (mm)', fontsize=14)
    plt.ylabel(f'risetime (ns)', fontsize=14)
    # plt.title(f'Risetime VS Radius \nfrom distribution', fontsize=14)
    plt.title(f'Risetime VS Radius for Background \nfrom distribution', fontsize=14)
    plt.setp(ax.get_xticklabels(), fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)
    plt.legend(fontsize=12, bbox_to_anchor=(1.04,1), borderaxespad=0)
    plt.tight_layout()
        
    if savefig:
        plt.savefig(f'./plots/{campaign}risetimes/bkg_rt_all_dist_vs_rad.png', dpi=200)
        plt.savefig(f'./plots/{campaign}risetimes/bkg_rt_all_dist_vs_rad.pdf', dpi=200)
        # plt.savefig(f'./plots/{campaign}risetimes/new_rt_all_dist_vs_rad.png', dpi=200)
    
    plt.clf()
    plt.close()
    
    
    # now for superpulses
    # 60 keV
    fig, ax = plt.subplots()
    for (tp, n) in zip(tp_list, range(len(tp_list))):
        plt.errorbar(rad_arr, tp_values[n], yerr=tp_full_err_arr[n], marker = markers[n], c=col[n], 
                     ls='none', label=f'{tp}')
    plt.xlabel('Radial Position (mm)', fontsize=14)
    plt.ylabel(f'risetime (ns)', fontsize=14)
    # plt.title(f'Risetime VS Radius \nfrom Superpulse', fontsize=14)
    plt.title(f'Risetime VS Radius for 60 keV \nfrom Superpulse', fontsize=14)
    plt.setp(ax.get_xticklabels(), fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)
    plt.legend(fontsize=12, bbox_to_anchor=(1.04,1), borderaxespad=0)
    plt.tight_layout()
        
    if savefig:
        plt.savefig(f'./plots/{campaign}risetimes/rt_all_vs_rad.png', dpi=200)
        plt.savefig(f'./plots/{campaign}risetimes/rt_all_vs_rad.pdf', dpi=200)
        # plt.savefig(f'./plots/{campaign}risetimes/new_rt_all_vs_rad.png', dpi=200)
    
    plt.clf()
    plt.close()
    
    # background 
    fig, ax = plt.subplots()
    for (tp, n) in zip(tp_list, range(len(tp_list))):
        plt.errorbar(rad_arr, bkg_tp_values[n], yerr=bkg_tp_full_err_arr[n], marker = markers[n], c=col[n], 
                     ls='none', label=f'{tp}')
    plt.xlabel('Radial Position (mm)', fontsize=14)
    plt.ylabel(f'risetime (ns)', fontsize=14)
    # plt.title(f'Risetime VS Radius \nfrom Superpulse', fontsize=14)
    plt.title(f'Risetime VS Radius for Background \nfrom Superpulse', fontsize=14)
    plt.setp(ax.get_xticklabels(), fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)
    plt.legend(fontsize=12, bbox_to_anchor=(1.04,1), borderaxespad=0)
    plt.tight_layout()
        
    if savefig:
        plt.savefig(f'./plots/{campaign}risetimes/bkg_rt_all_vs_rad.png', dpi=200)
        plt.savefig(f'./plots/{campaign}risetimes/bkg_rt_all_vs_rad.pdf', dpi=200)
        # plt.savefig(f'./plots/{campaign}risetimes/new_rt_all_vs_rad.png', dpi=200)
    
    plt.clf()
    plt.close()
    

def plot_superpulses(campaign, savefig=True, inset=False):

    # f_superpulse = './data/normScan/superpulses_60keV_allRuns.hdf5'
    # f_superpulse = './data/normScan/new_superpulses_60keV_allRuns.hdf5'
    f_superpulse = f'./data/normScan/superpulses_2sig_60keV_allRuns.hdf5'
    # f_superpulse = f'./data/normScan/superpulses_1sig_60keV_allRuns.hdf5'
    data_superpulse = pd.read_hdf(f_superpulse, key = '/superpulses')
    
    campaign+='2sig/'
    
    
    
    wf_arr = np.array(data_superpulse['pure_60_pz'])
    bkg_wf_arr = np.array(data_superpulse['bkg_pz'])
    bkg_wf = bkg_wf_arr[0]
    times = np.array(data_superpulse['samples'][0])
    rad_arr = np.array(data_superpulse['radius'])
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax = plt.axes()

    # set up colorbar to plot waveforms of different radii
    c = np.arange(0, len(wf_arr)+1)
    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.tab10)
    cmap.set_array([])

    for n in range(len(wf_arr)):
        ax.plot(times, wf_arr[n], c=cmap.to_rgba(n), label = f'{rad_arr[n]} mm')

    ax.plot(times, bkg_wf, c=cmap.to_rgba(5), label = f'Bkg')
    cb = fig.colorbar(cmap, spacing='proportional', ticks=range(len(rad_arr)+1), 
                      boundaries=np.arange(-0.5, len(rad_arr)+1.5))

    labels = np.append(rad_arr, 'Bkg')
    cb.ax.set_yticklabels(labels)
    cb.set_label("Radius (mm)", ha = 'right', va='center', rotation=270, fontsize=40, labelpad=20.) #20
    cb.ax.tick_params(labelsize=36) #18
    plt.locator_params(axis='x', nbins=6) # did this for my dissertation plots b/c needed larger 
                                          # font size and xticks became crowded

    plt.title(f'60 keV superpulses (PZ-corrected) \nnormal incidence', fontsize=40) #20

    plt.setp(ax.get_xticklabels(), fontsize=32) #16
    plt.setp(ax.get_yticklabels(), fontsize=32)
    plt.xlabel('clock cycles', fontsize=40) #20
    plt.ylabel('normalized ADU', fontsize=40)
    
    
    
    if savefig:
        plt.savefig(f'./plots/{campaign}waveforms/superpulses.png', dpi=200)
        plt.savefig(f'./plots/{campaign}waveforms/superpulses.pdf', dpi=200)
    
    plt.xlim(3700, 3900)
    
    if savefig:
        plt.savefig(f'./plots/{campaign}waveforms/superpulses_riseTail.png', dpi=200)
        plt.savefig(f'./plots/{campaign}waveforms/superpulses_riseTail.pdf', dpi=200)
        
    plt.xlim(3725, 3825)
    plt.ylim(-0.015, 0.6)
    if savefig:
        plt.savefig(f'./plots/{campaign}/waveforms/superpulses_rise.png', dpi=200)
        plt.savefig(f'./plots/{campaign}/waveforms/superpulses_rise.pdf', dpi=200)
        
    plt.xlim(3800, 4385)
    plt.ylim(0.91, 1.02)
    if savefig:
        plt.savefig(f'./plots/{campaign}waveforms/superpulses_tail.png', dpi=200)
        plt.savefig(f'./plots/{campaign}waveforms/superpulses_tail.pdf', dpi=200)
        
    plt.xlim(0, 3825)
    plt.ylim(-0.015, 0.1)
    if savefig:
        plt.savefig(f'./plots/{campaign}waveforms/superpulses_bl.png', dpi=200)
        plt.savefig(f'./plots/{campaign}waveforms/superpulses_bl.pdf', dpi=200)
        
    plt.xlim(3800, 8000)
    plt.ylim(0.91, 01.02)
    if savefig:
        plt.savefig(f'./plots/{campaign}waveforms/superpulses_fullTail.png', dpi=200)
        plt.savefig(f'./plots/{campaign}waveforms/superpulses_fullTail.pdf', dpi=200)
    

    
    if inset:
        ax = plt.gca()  # get the current axes
        ax.relim()      # make sure all the data fits
        ax.autoscale()
        
        # ____________________________________________
        # inset for risetime of WF
        axins = ax.inset_axes([200, 0.4, 3500,  0.6], transform=ax.transData) # [x_start, y_start, x_length, y_length]


        # sub region of the original image
        x1, x2, y1, y2 = 3725, 3825, -0.015, 0.6
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)

        for n in range(len(wf_arr)):
            axins.plot(times, wf_arr[n], c=cmap.to_rgba(n), label = f'{rad_arr[n]} mm')
        axins.plot(times, bkg_wf, c=cmap.to_rgba(5), label = f'Bkg')

        ax.indicate_inset_zoom(axins, edgecolor="black")

        # ____________________________________________
        # inset for top of WF
        axins2 = ax.inset_axes([5000, 0.1, 3000,  0.6], transform=ax.transData) # [x_start, y_start, x_length, y_length]


        # sub region of the original image
        x11, x22, y11, y22 = 3825, 4385, 0.9, 1.05
        axins2.set_xlim(x11, x22)
        axins2.set_ylim(y11, y22)

        for n in range(len(wf_arr)):
            axins2.plot(times, wf_arr[n], c=cmap.to_rgba(n), label = f'{rad_arr[n]} mm')
        axins2.plot(times, bkg_wf, c=cmap.to_rgba(5), label = f'Bkg')

        ax.indicate_inset_zoom(axins2, edgecolor="black")
        
        if savefig:
            plt.savefig(f'./plots/{campaign}waveforms/inset_superpulses.png', dpi=200)
            plt.savefig(f'./plots/{campaign}waveforms/inset_superpulses.pdf', dpi=200)
            
        plt.clf()
        plt.close()
        
def plot_bkgSub_superpulses(campaign, savefig=True):

    # f_superpulse = './data/normScan/superpulses_60keV_allRuns.hdf5'
    # f_superpulse = './data/normScan/new_superpulses_60keV_allRuns.hdf5'
    f_superpulse = f'./data/normScan/superpulses_2sig_60keV_allRuns.hdf5'
    # f_superpulse = f'./data/normScan/superpulses_1sig_60keV_allRuns.hdf5'
    data_superpulse = pd.read_hdf(f_superpulse, key = '/superpulses')
    
    campaign+='2sig/'
    
    
    
    peak_wf_raw = np.array(data_superpulse['bkg_and_60_raw'][0])
    bkg_wf_raw = np.array(data_superpulse['bkg_raw'][0]) 
    signal_wf_raw = np.array(data_superpulse['pure_60_raw'][0])
    
    peak_wf = cage_utils.notchFilter_SIS3302(peak_wf_raw, Q=20)
    bkg_wf = cage_utils.notchFilter_SIS3302(bkg_wf_raw, Q=20)
    signal_wf = cage_utils.notchFilter_SIS3302(signal_wf_raw, Q=20)

    times = np.array(data_superpulse['samples'][0])
    rad_arr = np.array(data_superpulse['radius'])
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax = plt.axes()
    
    ax.plot(times, peak_wf, c='b', label = f'Peak')

    ax.plot(times, bkg_wf, c='r', label = f'Background')
    
    ax.plot(times, signal_wf, c='g', label = f'Signal')

    plt.title(f'60 keV superpulse sideband subtraction \nwith notch filtering', fontsize=30) #20

    plt.setp(ax.get_xticklabels(), fontsize=30) #16
    plt.setp(ax.get_yticklabels(), fontsize=30)
    plt.xlabel('clock cycles', fontsize=30) #20
    plt.ylabel('ADU', fontsize=30)
    
    plt.legend(fontsize=30)
    
    
    if savefig:
        plt.savefig(f'./plots/{campaign}waveforms/superpulses_bkgSub_demo.png', dpi=200)
        plt.savefig(f'./plots/{campaign}waveforms/superpulses_bkgSub_demo.pdf', dpi=200)
        
    plt.xlim(3700, 3900)
    
    if savefig:
        plt.savefig(f'./plots/{campaign}waveforms/superpulses_bkgSub_demo_riseTail.png', dpi=200)
        plt.savefig(f'./plots/{campaign}waveforms/superpulses_bkgSub_demo_riseTail.pdf', dpi=200)

if __name__=="__main__":
    main()
