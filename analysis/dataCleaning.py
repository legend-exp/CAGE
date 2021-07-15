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
    runs = [60, 64, 66, 70, 72] # alpha runs for dsp_id = 2
    # runs = [62, 68, 74] #bkg runs for dsp_id = 2
    # campaign = 'angleScan/'
    campaign = 'new_normScan/centeringScans/'
    
    run=95

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

    # n_minus_1(run, campaign, df_raw, dg, runtype, rt_min, radius, angle_det, rotary, cut_keys_raw)
    allCuts(run, campaign, df_raw, dg, runtype, rt_min, radius, angle_det, rotary, cut_keys_raw)

def n_minus_1(run, campaign, df, dg, runtype, rt_min, radius, angle_det, rotary, cut_keys):

    with open('./cuts.json') as f:
        cuts = json.load(f)
    
    e_res_const = [0., 0., 0.]
    e_res_const[0] = cuts[str(run)]['e_res_const0']
    e_res_const[1] = cuts[str(run)]['e_res_const1']
    e_res_const[2] = cuts[str(run)]['e_res_const2']
        
    bl_cut_lo_raw = cuts[str(run)]['bl_cut_lo_raw']
    bl_cut_hi_raw = cuts[str(run)]['bl_cut_hi_raw'] 
    bl_slope_lo_raw = cuts[str(run)]['bl_slope_lo_raw']
    bl_slope_hi_raw = cuts[str(run)]['bl_slope_hi_raw']
    bl_sig_lo_raw = cuts[str(run)]['bl_sig_lo_raw']
    bl_sig_hi_raw = cuts[str(run)]['bl_sig_hi_raw']
    ftp_max_lo_raw = cuts[str(run)]['ftp_max_lo_raw']
    ftp_max_hi_raw = cuts[str(run)]['ftp_max_hi_raw'] 
    wf_max_fit_const = cuts[str(run)]['wf_max_fit_const']
    wf_max_fit_offset = cuts[str(run)]['wf_max_fit_offset']

    df = df.query(cuts[str(run)]['muon_cut']).copy()
    df_cut = df

    total_counts = len(df)
    print(f'total counts: {total_counts}')

    for cut_out in cut_keys:
        df_cut = df
        cut_set = cut_keys - set([cut_out])
        cut_full = " and ".join([cuts[str(run)][c] for c in cut_keys])
        print(f'Leaving out {cut_out}. \nfull cut: {cut_full}\n')
        
        #have to apply cuts individually instead of using `cut_full` because the total cut string is too long for the query :'(
        for cut in cut_set:
            print(f'applying cut: {cut}')
            df_cut = df_cut.query((cuts[str(run)][cut])).copy()
            cut_counts = len(df.query((cuts[str(run)][cut])).copy())
            percent_surviving = (cut_counts/total_counts)*100.
            print(f'Percentage surviving {cut} cut: {percent_surviving:.2f}')
        
        cut_counts_total = len(df_cut)
        percent_surviving_total = (cut_counts_total/total_counts)*100.
        print(f'Percentage surviving cuts: {percent_surviving_total:.2f}')
#         exit()

        # ____________baseline mean________________________________________

        fig, ax = plt.subplots()
#         suptitle = f'Run {run}; All cuts except: {cut_out}'
        suptitle = f'Run {run}; All cuts except: {cut_out}\n{percent_surviving_total:.2f}% surviving cuts'
        fig.suptitle(suptitle, horizontalalignment='center', fontsize=14)
        
        blo, bhi, bpb = 9000,9400, 1
        nbx = int((bhi-blo)/bpb)


        bl_hist, bins = np.histogram(df_cut['bl'], bins=nbx,
                range=[blo, bhi])
        bl_hist_raw, bins = np.histogram(df['bl'], bins=nbx,
                range=[blo, bhi])

        
        plt.semilogy(bins[1:], bl_hist_raw, c='k', alpha=0.3, ds='steps', lw=1., label='before cuts')
        plt.semilogy(bins[1:], bl_hist, ds='steps', c='b', lw=1, label = 'after cuts')
        
        plt.axvline(bl_cut_lo_raw, c='r', lw=1, label='95% cut lines')
        plt.axvline(bl_cut_hi_raw, c='r', lw=1)

        plt.xlabel('bl', fontsize=14)
        plt.ylabel('counts', fontsize=14)

#         plt.title(f'Baseline Mean \n{percent_surviving_total:.2f}% surviving cuts', fontsize = 14)
        plt.title(f'Baseline Mean', fontsize = 14)

        plt.setp(ax.get_xticklabels(), fontsize=12)
        plt.setp(ax.get_yticklabels(), fontsize=12)

        ax.text(0.05, 0.75, f'r = {radius} mm \ntheta = {angle_det} deg \nruntime {rt_min:.2f}', verticalalignment='bottom',
                horizontalalignment='left', transform=ax.transAxes, color='black', fontsize=12, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10}) #0.1, 0.75, 
        
        plt.legend(loc='center left')
        
        plt.tight_layout()

        plt.savefig(f'./plots/{campaign}N_minus_1/raw/{str(run)}/except_{cut_out}_bl_mean_raw.png', dpi=200)
        plt.clf()
        plt.close()
        
#         exit()


        # ____________baseline slope________________________________________

        fig, ax = plt.subplots()
        fig.suptitle(suptitle, horizontalalignment='center', fontsize=14)
        
        blo, bhi, bpb = -10., 10., 0.005
        nbx = int((bhi-blo)/bpb)

        bl_hist, bins = np.histogram(df_cut['bl_slope'], bins=nbx,range=[blo, bhi])
        bl_hist_raw, bins = np.histogram(df['bl_slope'], bins=nbx,range=[blo, bhi])

        plt.semilogy(bins[1:], bl_hist_raw, ds='steps', c='k', alpha=0.3, lw=1, label='before cuts')
        plt.semilogy(bins[1:], bl_hist, ds='steps', c='b', lw=1, label = 'after cuts')
        
        
        plt.axvline(bl_slope_lo_raw, c='r', lw=1, label = '95% cut lines')
        plt.axvline(bl_slope_hi_raw, c='r', lw=1)


        plt.xlabel('bl_slope', fontsize=14)
        plt.ylabel('counts', fontsize=14)

        plt.title(f'Baseline Slope', fontsize = 14)

        plt.setp(ax.get_xticklabels(), fontsize=12)
        plt.setp(ax.get_yticklabels(), fontsize=12)
        

        ax.text(0.05, 0.75, f'r = {radius} mm \ntheta = {angle_det} deg \nruntime {rt_min:.2f}', verticalalignment='bottom',
                horizontalalignment='left', transform=ax.transAxes, color='black', fontsize=12, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})
        
        plt.legend()
        
        plt.tight_layout()

        plt.savefig(f'./plots/{campaign}dataCleaning/N_minus_1/raw/{str(run)}/except_{cut_out}_bl_slope_raw.png', dpi=200)
        plt.clf()
        plt.close()

        # ____________baseline sigma________________________________________

        fig, ax = plt.subplots()
        fig.suptitle(suptitle, horizontalalignment='center', fontsize=14)

        blo, bhi, bpb = 2., 12., 0.005
        nbx = int((bhi-blo)/bpb)

        bl_hist, bins = np.histogram(df_cut['bl_sig'], bins=nbx, range=[blo, bhi])
        bl_hist_raw, bins = np.histogram(df['bl_sig'], bins=nbx, range=[blo, bhi])

        plt.semilogy(bins[1:], bl_hist_raw, ds='steps', c='k', alpha=0.3, lw=1, label='before cuts')
        plt.semilogy(bins[1:], bl_hist, ds='steps', c='b', lw=1, label = 'after cuts')
        
        
        plt.axvline(bl_sig_lo_raw, c='r', lw=1, label = '95% cut lines')
        plt.axvline(bl_sig_hi_raw, c='r', lw=1)

        plt.xlabel('bl_sigma', fontsize=14)
        plt.ylabel('counts', fontsize=14)

        plt.title(f'Baseline Sigma', fontsize = 14)

        plt.setp(ax.get_xticklabels(), fontsize=12)
        plt.setp(ax.get_yticklabels(), fontsize=12)

        ax.text(0.9, 0.75, f'r = {radius} mm \ntheta = {angle_det} deg \nruntime {rt_min:.2f}', verticalalignment='bottom',
            horizontalalignment='right', transform=ax.transAxes, color='black', fontsize=12, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})
        
        plt.legend(loc='center right')
        
        plt.tight_layout()


        plt.savefig(f'./plots/{campaign}dataCleaning/N_minus_1/raw/{str(run)}/except_{cut_out}_bl_sig_raw.png', dpi=200)
        plt.clf()
        plt.close()

        # ____________trapEftp/trapEmax________________________________________

        fig, ax = plt.subplots()
        fig.suptitle(suptitle, horizontalalignment='center', fontsize=14)

        elo, ehi = 0.925, 1.01
        e_bins = int((ehi - elo )/0.001)

        ftp_max_hist, bins = np.histogram(df_cut['ftp_max'], bins=nbx, range=[elo, ehi])
        ftp_max_hist_raw, bins = np.histogram(df['ftp_max'], bins=nbx, range=[elo, ehi])

        plt.semilogy(bins[1:], ftp_max_hist_raw, ds='steps', c='k', alpha=0.3, lw=1, label='before cuts')
        plt.semilogy(bins[1:], ftp_max_hist, ds='steps', c='b', lw=1, label = 'after cuts')
        
        plt.axvline(ftp_max_lo_raw, c='r', lw=1, label='95% cut lines')
        plt.axvline(ftp_max_hi_raw, c='r', lw=1)


        plt.xlabel('trapEftp/trapEmax', fontsize=14)
        plt.ylabel('counts', fontsize=14)

        plt.title(f'trapEftp/trapEmax', fontsize = 14)

        plt.setp(ax.get_xticklabels(), fontsize=12)
        plt.setp(ax.get_yticklabels(), fontsize=12)

        ax.text(0.1, 0.75, f'r = {radius} mm \ntheta = {angle_det} deg \nruntime {rt_min:.2f}', verticalalignment='bottom',
            horizontalalignment='left', transform=ax.transAxes, color='black', fontsize=12, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})
        
        plt.legend(loc='center left')
        
        plt.tight_layout()

        plt.savefig(f'./plots/{campaign}dataCleaning/N_minus_1/raw/{str(run)}/except_{cut_out}_ftp_max_raw.png', dpi=200)
        plt.clf()
        plt.close()

        # ____________wf_maxVtrapEftp_cal________________________________________

        fig, ax = plt.subplots()
        fig.suptitle(suptitle, horizontalalignment='center', fontsize=14)
        
        elo, ehi, epb = 0, 5500, 1
        e_bins = 2000 #int((ehi-elo)/epb)
        wflo, wfhi = 0, 15000
        wf_bins = 2000
        wf_maxVEnergy, xedges, yedges = np.histogram2d(df_cut['wf_max'], df_cut['trapEftp_cal'], bins=[wf_bins, e_bins], range=([wflo, wfhi], [elo, ehi]))
        X, Y = np.mgrid[wflo:wfhi:wf_bins*1j, elo:ehi:e_bins*1j]


        pcm = plt.pcolormesh(X, Y, wf_maxVEnergy,norm=LogNorm())
        cb = plt.colorbar()
        cb.set_label("counts", ha = 'right', va='center', rotation=270, fontsize=14)
        cb.ax.tick_params(labelsize=12)

        ax.text(0.1, 0.75,  f'r = {radius} mm \ntheta = {angle_det} deg \nruntime {rt_min:.2f}', verticalalignment='bottom',
            horizontalalignment='left', transform=ax.transAxes, color='black', fontsize=12, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})
        
        # note: plotting the fit lines is only reliable if you used the same binning as when the fit was done!
        en_bin_centers = pgh.get_bin_centers(xedges)
        cal_en_bin_centers = pgh.get_bin_centers(yedges)
        
        z = (wf_max_fit_const*en_bin_centers + wf_max_fit_offset + 2.*np.sqrt(e_res_const[0]+e_res_const[1]*cal_en_bin_centers +
            (e_res_const[2]*cal_en_bin_centers**2))) 
        plt.plot(en_bin_centers, z, 'r', lw = 0.7, label= 'cut lines')
        
        w = (wf_max_fit_const*en_bin_centers + wf_max_fit_offset - 2.*np.sqrt(e_res_const[0]+e_res_const[1]*cal_en_bin_centers +
            e_res_const[2]*cal_en_bin_centers**2))
        plt.plot(en_bin_centers, w, 'r', lw=0.7)

        ax.set_xlabel('wf_max', fontsize=14)
        ax.set_ylabel('trapEftp_cal (keV)', fontsize=14)
        plt.setp(ax.get_xticklabels(), fontsize=12)
        plt.setp(ax.get_yticklabels(), fontsize=12)
        plt.title(f'wf_max vs Energy', horizontalalignment='center', fontsize=14)
        
        plt.legend(loc='lower right')
        
        plt.tight_layout()


        plt.ylim(0, 300)
        plt.xlim(0, 800)

        plt.savefig(f'./plots/{campaign}dataCleaning/N_minus_1/raw/{str(run)}/except_{cut_out}_wf_max_raw_lowE.png', dpi=200)

        plt.ylim(1200, 1550)
        plt.xlim(3300, 4300)

        plt.savefig(f'./plots/{campaign}dataCleaning/N_minus_1/raw/{str(run)}/except_{cut_out}_wf_max_raw_1460.png', dpi=200)


        plt.ylim(2400, 2750)
        plt.xlim(6600, 8000)

        plt.savefig(f'./plots/{campaign}dataCleaning/N_minus_1/raw/{str(run)}/except_{cut_out}_wf_max_raw_2615.png', dpi=200)
        plt.clf()
        plt.close()



        # ____________60 keV with fit________________________________________

        pgfenergy_hist, pgfebins, evars = pgh.get_hist(df_cut['trapEftp_cal'], bins=50, range=[54, 65])
        raw_pgfenergy_hist, pgfebins, evars = pgh.get_hist(df['trapEftp_cal'], bins=50, range=[54, 65])#range=[54, 65]
        pars, cov = pgf.gauss_mode_width_max(pgfenergy_hist, pgfebins, evars)
        mode = pars[0]
        width = pars[1]
        amp = pars[2]
        print(f'mode: {mode}')
        print(f'width: {width}')
        print(f'amp: {amp}')


        e_pars, ecov = pgf.fit_hist(cage_utils.gauss_fit_func, pgfenergy_hist, pgfebins, evars, guess = (amp, mode, width, 1))

        mean_fit = e_pars[1]
        width_fit = e_pars[2]
        amp_fit = e_pars[0]
        const_fit = e_pars[3]

        fwhm = width_fit*2.355

        print(f'mean: {mean_fit}')
        print(f'width: {width_fit}')
        print(f'amp: {amp_fit}')
        print(f'C: {const_fit}')
        print(f'FWHM at 60 keV: {fwhm} \n{(fwhm/mean_fit)*100}%')

        fig, ax = plt.subplots()
        fig.suptitle(suptitle, horizontalalignment='center', fontsize=14)

        plt.plot(pgfebins[1:], cage_utils.gauss_fit_func(pgfebins[1:], *e_pars), c = 'r', lw=0.8, label='gaussian fit')
        plt.plot(pgfebins[1:], pgfenergy_hist, ds='steps', c='b', lw=1., label='after cuts')
        plt.plot(pgfebins[1:], raw_pgfenergy_hist, ds='steps', c='k', alpha=0.3, lw=1., label='before cuts')

        plt.xlabel('Energy (keV)', fontsize=14)
        plt.ylabel('counts', fontsize=14)

        plt.title(f'60 keV peak with gaussian fit', fontsize = 14)

        plt.setp(ax.get_xticklabels(), fontsize=12)
        plt.setp(ax.get_yticklabels(), fontsize=12)

        ax.text(0.05, 0.75,  f'r = {radius} mm \ntheta = {angle_det} deg \nruntime {rt_min:.2f}', verticalalignment='bottom',
            horizontalalignment='left', transform=ax.transAxes, color='black', fontsize=10, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 8})
        ax.text(0.95, 0.72,  f'mean: {mean_fit:.2f} \nsigma: {width_fit:.3f} \nFWHM at 60 keV: {fwhm:.2f} keV\n({(fwhm/mean_fit)*100:.2f}%)', verticalalignment='bottom',
            horizontalalignment='right', transform=ax.transAxes, color='black', fontsize=10, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 8})
        
        plt.legend(loc='center right')
        
        plt.tight_layout()

        plt.savefig(f'./plots/{campaign}dataCleaning/N_minus_1/raw/{str(run)}/except_{cut_out}_fit_60keV_raw.png', dpi=200)
        plt.clf()
        plt.close()


def allCuts(run, campaign, df, dg, runtype, rt_min, radius, angle_det, rotary, cut_keys):

    with open('./cuts.json') as f:
        cuts = json.load(f)
    
    e_res_const = [0., 0., 0.]
    e_res_const[0] = cuts[str(run)]['e_res_const0']
    e_res_const[1] = cuts[str(run)]['e_res_const1']
    e_res_const[2] = cuts[str(run)]['e_res_const2']
        
    bl_cut_lo_raw = cuts[str(run)]['bl_cut_lo_raw']
    bl_cut_hi_raw = cuts[str(run)]['bl_cut_hi_raw'] 
    bl_slope_lo_raw = cuts[str(run)]['bl_slope_lo_raw']
    bl_slope_hi_raw = cuts[str(run)]['bl_slope_hi_raw']
    bl_sig_lo_raw = cuts[str(run)]['bl_sig_lo_raw']
    bl_sig_hi_raw = cuts[str(run)]['bl_sig_hi_raw']
    ftp_max_lo_raw = cuts[str(run)]['ftp_max_lo_raw']
    ftp_max_hi_raw = cuts[str(run)]['ftp_max_hi_raw'] 
    wf_max_fit_const = cuts[str(run)]['wf_max_fit_const']
    wf_max_fit_offset = cuts[str(run)]['wf_max_fit_offset']

    df = df.query(cuts[str(run)]['muon_cut']).copy()
    df_cut = df

    total_counts = len(df)
    print(f'total counts: {total_counts}')

        
        #have to apply cuts individually instead of using `cut_full` because the total cut string is too long for the query :'(
    for cut in cut_keys:
        print(f'applying cut: {cut}')
        df_cut = df_cut.query((cuts[str(run)][cut])).copy()
        cut_counts = len(df.query((cuts[str(run)][cut])).copy())
        percent_surviving = (cut_counts/total_counts)*100.
        print(f'Percentage surviving {cut} cut: {percent_surviving:.2f}')
        
    cut_counts_total = len(df_cut)
    percent_surviving_total = (cut_counts_total/total_counts)*100.
    print(f'Percentage surviving cuts: {percent_surviving_total:.2f}')
#     exit()

    # ____________baseline mean________________________________________

    fig, ax = plt.subplots()
    suptitle = f'Run {run}; All cuts applied\n{percent_surviving_total:.2f}% surviving cuts'
    fig.suptitle(suptitle, horizontalalignment='center', fontsize=14)
        
    blo, bhi, bpb = 9000,9400, 1
    nbx = int((bhi-blo)/bpb)


    bl_hist, bins = np.histogram(df_cut['bl'], bins=nbx,
                range=[blo, bhi])
    bl_hist_raw, bins = np.histogram(df['bl'], bins=nbx,
                range=[blo, bhi])

        
    plt.semilogy(bins[1:], bl_hist_raw, c='k', alpha=0.3, ds='steps', lw=1., label='before cuts')
    plt.semilogy(bins[1:], bl_hist, ds='steps', c='b', lw=1, label = 'after cuts')
        
    plt.axvline(bl_cut_lo_raw, c='r', lw=1, label='95% cut lines')
    plt.axvline(bl_cut_hi_raw, c='r', lw=1)

    plt.xlabel('bl', fontsize=14)
    plt.ylabel('counts', fontsize=14)
    
    plt.title(f'Baseline Mean', fontsize = 14)

    plt.setp(ax.get_xticklabels(), fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)

    ax.text(0.05, 0.75, f'r = {radius} mm \ntheta = {angle_det} deg \nruntime {rt_min:.2f}', verticalalignment='bottom',
                horizontalalignment='left', transform=ax.transAxes, color='black', fontsize=12, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10}) #0.1, 0.75, 
        
    plt.legend(loc='center left')
        
    plt.tight_layout()

    plt.savefig(f'./plots/{campaign}dataCleaning/allCuts/{str(run)}/allCuts_bl_mean_raw.png', dpi=200)
    plt.clf()
    plt.close()
        
#         exit()


    # ____________baseline slope________________________________________

    fig, ax = plt.subplots()
    fig.suptitle(suptitle, horizontalalignment='center', fontsize=14)
        
    blo, bhi, bpb = -10., 10., 0.005
    nbx = int((bhi-blo)/bpb)

    bl_hist, bins = np.histogram(df_cut['bl_slope'], bins=nbx,range=[blo, bhi])
    bl_hist_raw, bins = np.histogram(df['bl_slope'], bins=nbx,range=[blo, bhi])

    plt.semilogy(bins[1:], bl_hist_raw, ds='steps', c='k', alpha=0.3, lw=1, label='before cuts')
    plt.semilogy(bins[1:], bl_hist, ds='steps', c='b', lw=1, label = 'after cuts')
        
        
    plt.axvline(bl_slope_lo_raw, c='r', lw=1, label = '95% cut lines')
    plt.axvline(bl_slope_hi_raw, c='r', lw=1)


    plt.xlabel('bl_slope', fontsize=14)
    plt.ylabel('counts', fontsize=14)

    plt.title(f'Baseline Slope', fontsize = 14)

    plt.setp(ax.get_xticklabels(), fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)
        

    ax.text(0.04, 0.75, f'r = {radius} mm \ntheta = {angle_det} deg \nruntime {rt_min:.2f}', verticalalignment='bottom',
                horizontalalignment='left', transform=ax.transAxes, color='black', fontsize=12, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})
        
    plt.legend(loc='upper right')
        
    plt.tight_layout()

    plt.savefig(f'./plots/{campaign}dataCleaning/allCuts/{str(run)}/allCuts_bl_slope_raw.png', dpi=200)
    plt.clf()
    plt.close()

    # ____________baseline sigma________________________________________

    fig, ax = plt.subplots()
    fig.suptitle(suptitle, horizontalalignment='center', fontsize=14)

    blo, bhi, bpb = 2., 12., 0.005
    nbx = int((bhi-blo)/bpb)

    bl_hist, bins = np.histogram(df_cut['bl_sig'], bins=nbx, range=[blo, bhi])
    bl_hist_raw, bins = np.histogram(df['bl_sig'], bins=nbx, range=[blo, bhi])

    plt.semilogy(bins[1:], bl_hist_raw, ds='steps', c='k', alpha=0.3, lw=1, label='before cuts')
    plt.semilogy(bins[1:], bl_hist, ds='steps', c='b', lw=1, label = 'after cuts')
        
        
    plt.axvline(bl_sig_lo_raw, c='r', lw=1, label = '95% cut lines')
    plt.axvline(bl_sig_hi_raw, c='r', lw=1)

    plt.xlabel('bl_sigma', fontsize=14)
    plt.ylabel('counts', fontsize=14)

    plt.title(f'Baseline Sigma', fontsize = 14)

    plt.setp(ax.get_xticklabels(), fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)

    ax.text(0.9, 0.75, f'r = {radius} mm \ntheta = {angle_det} deg \nruntime {rt_min:.2f}', verticalalignment='bottom',
            horizontalalignment='right', transform=ax.transAxes, color='black', fontsize=12, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})
        
    plt.legend(loc='center right')
        
    plt.tight_layout()


    plt.savefig(f'./plots/{campaign}dataCleaning/allCuts/{str(run)}/allCuts_bl_sig_raw.png', dpi=200)
    plt.clf()
    plt.close()

    # ____________trapEftp/trapEmax________________________________________

    fig, ax = plt.subplots()
    fig.suptitle(suptitle, horizontalalignment='center', fontsize=14)

    elo, ehi = 0.925, 1.01
    e_bins = int((ehi - elo )/0.001)

    ftp_max_hist, bins = np.histogram(df_cut['ftp_max'], bins=nbx, range=[elo, ehi])
    ftp_max_hist_raw, bins = np.histogram(df['ftp_max'], bins=nbx, range=[elo, ehi])

    plt.semilogy(bins[1:], ftp_max_hist_raw, ds='steps', c='k', alpha=0.3, lw=1, label='before cuts')
    plt.semilogy(bins[1:], ftp_max_hist, ds='steps', c='b', lw=1, label = 'after cuts')
        
    plt.axvline(ftp_max_lo_raw, c='r', lw=1, label='95% cut lines')
    plt.axvline(ftp_max_hi_raw, c='r', lw=1)


    plt.xlabel('trapEftp/trapEmax', fontsize=14)
    plt.ylabel('counts', fontsize=14)

    plt.title(f'trapEftp/trapEmax', fontsize = 14)

    plt.setp(ax.get_xticklabels(), fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)

    ax.text(0.1, 0.75, f'r = {radius} mm \ntheta = {angle_det} deg \nruntime {rt_min:.2f}', verticalalignment='bottom',
            horizontalalignment='left', transform=ax.transAxes, color='black', fontsize=12, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})
        
    plt.legend(loc='center left')
        
    plt.tight_layout()

    plt.savefig(f'./plots/{campaign}dataCleaning/allCuts/{str(run)}/allCuts_ftp_max_raw.png', dpi=200)
    plt.clf()
    plt.close()

    # ____________wf_maxVtrapEftp_cal________________________________________

    fig, ax = plt.subplots()
    fig.suptitle(suptitle, horizontalalignment='center', fontsize=14)
        
    elo, ehi, epb = 0, 5500, 1
    e_bins = 2000 #int((ehi-elo)/epb)
    wflo, wfhi = 0, 15000
    wf_bins = 2000
    wf_maxVEnergy, xedges, yedges = np.histogram2d(df_cut['wf_max'], df_cut['trapEftp_cal'], bins=[wf_bins, e_bins], range=([wflo, wfhi], [elo, ehi]))
    X, Y = np.mgrid[wflo:wfhi:wf_bins*1j, elo:ehi:e_bins*1j]


    pcm = plt.pcolormesh(X, Y, wf_maxVEnergy,norm=LogNorm())
    cb = plt.colorbar()
    cb.set_label("counts", ha = 'right', va='center', rotation=270, fontsize=14)
    cb.ax.tick_params(labelsize=12)

    ax.text(0.1, 0.75,  f'r = {radius} mm \ntheta = {angle_det} deg \nruntime {rt_min:.2f}', verticalalignment='bottom',
            horizontalalignment='left', transform=ax.transAxes, color='black', fontsize=12, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})
        
    # note: plotting the fit lines is only reliable if you used the same binning as when the fit was done!
    en_bin_centers = pgh.get_bin_centers(xedges)
    cal_en_bin_centers = pgh.get_bin_centers(yedges)
        
    z = (wf_max_fit_const*en_bin_centers + wf_max_fit_offset + 2.*np.sqrt(e_res_const[0]+e_res_const[1]*cal_en_bin_centers +
            (e_res_const[2]*cal_en_bin_centers**2))) 
    plt.plot(en_bin_centers, z, 'r', lw = 0.7, label= 'cut lines')
        
    w = (wf_max_fit_const*en_bin_centers + wf_max_fit_offset - 2.*np.sqrt(e_res_const[0]+e_res_const[1]*cal_en_bin_centers +
            e_res_const[2]*cal_en_bin_centers**2))
    plt.plot(en_bin_centers, w, 'r', lw=0.7)

    ax.set_xlabel('wf_max', fontsize=14)
    ax.set_ylabel('trapEftp_cal (keV)', fontsize=14)
    plt.setp(ax.get_xticklabels(), fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)
    plt.title(f'wf_max vs Energy', horizontalalignment='center', fontsize=14)
        
    plt.legend(loc='lower right')
        
    plt.tight_layout()


    plt.ylim(0, 300)
    plt.xlim(0, 800)

    plt.savefig(f'./plots/{campaign}dataCleaning/allCuts/{str(run)}/allCuts_wf_max_raw_lowE.png', dpi=200)

    plt.ylim(1200, 1550)
    plt.xlim(3300, 4300)

    plt.savefig(f'./plots/{campaign}dataCleaning/allCuts/{str(run)}/allCuts_wf_max_raw_1460.png', dpi=200)


    plt.ylim(2400, 2750)
    plt.xlim(6600, 8000)

    plt.savefig(f'./plots/{campaign}dataCleaning/allCuts/{str(run)}/allCuts_wf_max_raw_2615.png', dpi=200)
    plt.clf()
    plt.close()



    # ____________60 keV with fit________________________________________

    pgfenergy_hist, pgfebins, evars = pgh.get_hist(df_cut['trapEftp_cal'], bins=35, range=[54, 65])
    raw_pgfenergy_hist, pgfebins, evars = pgh.get_hist(df['trapEftp_cal'], bins=35, range=[54, 65])#range=[54, 65]
    pars, cov = pgf.gauss_mode_width_max(pgfenergy_hist, pgfebins, evars)
    mode = pars[0]
    width = pars[1]
    amp = pars[2]

    e_pars, ecov = pgf.fit_hist(cage_utils.gauss_fit_func, pgfenergy_hist, pgfebins, evars, guess = (amp, mode, width, 1))

    mean_fit = e_pars[1]
    width_fit = e_pars[2]
    amp_fit = e_pars[0]
    const_fit = e_pars[3]

    fwhm = width_fit*2.355

    print(f'mean: {mean_fit}')
    print(f'width: {width_fit}')
    print(f'amp: {amp_fit}')
    print(f'C: {const_fit}')
    print(f'FWHM at 60 keV: {fwhm} \n{(fwhm/mean_fit)*100}%')

    fig, ax = plt.subplots()
    fig.suptitle(suptitle, horizontalalignment='center', fontsize=14)

    plt.plot(pgfebins[1:], cage_utils.gauss_fit_func(pgfebins[1:], *e_pars), c = 'r', lw=0.8, label='gaussian fit')
    plt.plot(pgfebins[1:], pgfenergy_hist, ds='steps', c='b', lw=1., label='after cuts')
    plt.plot(pgfebins[1:], raw_pgfenergy_hist, ds='steps', c='k', alpha=0.3, lw=1., label='before cuts')

    plt.xlabel('Energy (keV)', fontsize=14)
    plt.ylabel('counts', fontsize=14)

    plt.title(f'60 keV peak with gaussian fit', fontsize = 14)

    plt.setp(ax.get_xticklabels(), fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)

    ax.text(0.05, 0.75,  f'r = {radius} mm \ntheta = {angle_det} deg \nruntime {rt_min:.2f}', verticalalignment='bottom',
            horizontalalignment='left', transform=ax.transAxes, color='black', fontsize=10, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 8})
    ax.text(0.95, 0.72,  f'mean: {mean_fit:.2f} \nsigma: {width_fit:.3f} \nFWHM at 60 keV: {fwhm:.2f} keV\n({(fwhm/mean_fit)*100:.2f}%)', verticalalignment='bottom',
            horizontalalignment='right', transform=ax.transAxes, color='black', fontsize=10, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 8})
        
    plt.legend(loc='center right')
        
    plt.tight_layout()

    plt.savefig(f'./plots/{campaign}dataCleaning/allCuts/{str(run)}/allCuts_fit_60keV_raw.png', dpi=200)
    plt.clf()
    plt.close()


if __name__=="__main__":
    main()
