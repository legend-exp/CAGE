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
    runs = [60, 64, 66, 70, 72] # alpha runs for dsp_id = 2
#     runs = [62, 68, 74] #bkg runs for dsp_id = 2
#     runs = [50]
#     alp_runs = [137, 143]
#     bkg_runs = [136, 136]
    # campaign = 'angleScan/'
    campaign = 'new_normScan/'

    user = True
    hit = True
    cal = True
    etype = 'trapEftp_cal'

    dsp_list = ['energy', 'trapEftp', 'trapEmax', 'trapEftp_cal', 'bl','bl_sig', 'bl_slope', 'AoE', 'dcr', "tp_0",
            "tp_02", "tp_05", "tp_10", "tp_20", 'tp_max', 'ToE', 'log_tail_fit_slope', 'wf_max', 'wf_argmax', 'trapE_argmax', 'lf_max']

    cut_keys = set('wf_max_cut', 'bl_mean_cut_raw', 'bl_mean_cut', 'bl_slope_cut_raw', 'bl_slope_cut',
            'bl_sig_cut_raw', 'bl_sig_cut', 'ftp_max_cut_raw', 'ftp_max_cut')

    df_raw, dg, runtype, rt_min, radius, angle_det, rotary = cage_utils.getDataFrame(run, user=user, hit=hit, cal=cal, dsp_list=dsp_list, lowE=lowE)

    n_minus_1(df_raw, dg, runtype, rt_min, radius, angle_det, rotary, cut_keys)

def n_minus_1(df, dg, runtype, rt_min, radius, angle_det, rotary, cut_keys):

    with open('./cuts.json') as f:
        cuts = json.load(f)

    df = df.query([cuts[str(run)]['muon_cut']).copy()

    total_counts = len(df)
    print(f'total counts: {total_counts}')

    for cut_out in cut_keys:
        cut_set = cut_keys - set([cut_out])
        cut_full = " and ".join([cuts[str(run)][c] for c in cut_keys])

        df_cut = df.query(cut_full).copy()
        cut_counts = len(df_cut)
        percent_surviving = (cut_counts/total_counts)*100.

        print(f'Leaving out {cut_out}. \nfull cut: {cut_full}\n')
        print(f'Percentage surviving cuts: {percent_surviving:.2f}')

        # ____________baseline mean________________________________________

        fig, ax = plt.subplots()
        suptitle = f'All cuts except {cut_out}\n{percent_surviving:.2f}% surviving cuts'
        blo, bhi, bpb = 9000,9400, 1
        nbx = int((bhi-blo)/bpb)


        bl_hist, bins = np.histogram(df['bl'], bins=nbx,
                range=[blo, bhi])

        plt.plot(bins[1:], bl_hist, ds='steps', c='b', lw=1)

        plt.xlabel('bl', fontsize=16)
        plt.ylabel('counts', fontsize=16)

        plt.title('Baseline Mean', fontsize = 16)

        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)

        ax.text(0.95, 0.83, f'r = {radius} mm \ntheta = {angle_det} deg \nruntime {rt_min:.2f}', verticalalignment='bottom',
                horizontalalignment='right', transform=ax.transAxes, color='black', fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

        plt.savefig(f'./plots/new_normScan/dataCleaning/N_minus_1/raw/except_{cut_out}_bl_mean_raw.png', dpi=200)
        plt.clf()
        plt.close()


        # ____________baseline slope________________________________________

        fig, ax = plt.subplots()
        suptitle = f'All cuts except {cut_out}\n{percent_surviving:.2f}% surviving cuts'
        blo, bhi, bpb = -10., 10., 0.005
        nbx = int((bhi-blo)/bpb)

        bl_hist, bins = np.histogram(df['bl_slope'], bins=nbx,range=[blo, bhi])

        plt.plot(bins[1:], bl_hist, ds='steps', c='b', lw=1)


        plt.xlabel('bl_slope', fontsize=16)
        plt.ylabel('counts', fontsize=16)

        plt.title('Baseline Slope', fontsize = 16)

        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)

        ax.text(0.95, 0.83, f'r = {radius} mm \ntheta = {angle_det} deg \nruntime {rt_min:.2f}', verticalalignment='bottom',
                horizontalalignment='right', transform=ax.transAxes, color='black', fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

        plt.savefig(f'./plots/new_normScan/dataCleaning/N_minus_1/raw/except_{cut_out}_bl_slope_raw.png', dpi=200)
        plt.clf()
        plt.close()

        # ____________baseline sigma________________________________________

        fig, ax = plt.subplots()
        suptitle = f'All cuts except {cut_out}\n{percent_surviving:.2f}% surviving cuts'

        blo, bhi, bpb = 2., 12., 0.005
        nbx = int((bhi-blo)/bpb)

        bl_hist, bins = np.histogram(df['bl_sig'], bins=nbx, range=[blo, bhi])

        plt.plot(bins[1:], bl_hist, ds='steps', c='b', lw=1)

        plt.xlabel('bl_sigma', fontsize=16)
        plt.ylabel('counts', fontsize=16)

        plt.title('Baseline Sigma', fontsize = 16)

        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)

        ax.text(0.95, 0.83, f'r = {radius} mm \ntheta = {angle_det} deg \nruntime {rt_min:.2f}', verticalalignment='bottom',
            horizontalalignment='right', transform=ax.transAxes, color='black', fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})


        plt.savefig(f'./plots/new_normScan/dataCleaning/N_minus_1/raw/except_{cut_out}_bl_sig_raw.png', dpi=200)
        plt.clf()
        plt.close()

        # ____________trapEftp/trapEmax________________________________________

        fig, ax = plt.subplots()
        suptitle = f'All cuts except {cut_out}\n{percent_surviving:.2f}% surviving cuts'

        elo, ehi = 0.925, 1.01
        e_bins = int((ehi - elo )/0.001)

        ftp_max_hist, bins = np.histogram(df['ftp_max'], bins=nbx, range=[elo, ehi])

        plt.plot(bins[1:], ftp_max_hist, ds='steps', c='b', lw=1)


        plt.xlabel('trapEftp/trapEmax', fontsize=16)
        plt.ylabel('counts', fontsize=16)

        plt.title('trapEftp/trapEmax before cuts\nwith 95% cut lines', fontsize = 16)

        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)

        ax.text(0.95, 0.83, f'r = {radius} mm \ntheta = {angle_det} deg \nruntime {rt_min:.2f}', position = (0.1, 0.85), transform=ax.transAxes, color='black', fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

        plt.savefig(f'./plots/new_normScan/dataCleaning/N_minus_1/raw/except_{cut_out}_ftp_max_raw.png', dpi=200)
        plt.clf()
        plt.close()

        # ____________wf_maxVtrapEftp_cal________________________________________

        fig, ax = plt.subplots()
        suptitle = f'All cuts except {cut_out}\n{percent_surviving:.2f}% surviving cuts'
        elo, ehi, epb = 0, 5500, 1
        e_bins = 2000 #int((ehi-elo)/epb)
        wflo, wfhi = 0, 15000
        wf_bins = 2000
        fig.suptitle(f'\nwf_max vs Energy', horizontalalignment='center', fontsize=16)
        wf_maxVEnergy, xedges, yedges = np.histogram2d(df['wf_max'], df['trapEftp_cal'], bins=[wf_bins, e_bins], range=([wflo, wfhi], [elo, ehi]))
        X, Y = np.mgrid[wflo:wfhi:wf_bins*1j, elo:ehi:e_bins*1j]


        pcm = plt.pcolormesh(X, Y, wf_maxVEnergy,norm=LogNorm())
        cb = plt.colorbar()
        cb.set_label("counts", ha = 'right', va='center', rotation=270, fontsize=14)
        cb.ax.tick_params(labelsize=14)

        ax.text(0.95, 0.83, f'r = {radius} mm \ntheta = {angle_det} deg \nruntime {rt_min:.2f}', position=(0.1, 0.85), transform=ax.transAxes, color='black', fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

        ax.set_xlabel('wf_max', fontsize=16)
        ax.set_ylabel('trapEftp_cal (keV)', fontsize=16)
        plt.setp(ax.get_xticklabels(), fontsize=12)
        plt.setp(ax.get_yticklabels(), fontsize=14)


        plt.ylim(0, 300)
        plt.xlim(0, 800)

        plt.savefig(f'./plots/new_normScan/dataCleaning/N_minus_1/raw/except_{cut_out}_wf_max_raw_lowE.png', dpi=200)

        plt.ylim(1200, 1550)
        plt.xlim(3300, 4300)

        plt.savefig(f'./plots/new_normScan/dataCleaning/N_minus_1/raw/except_{cut_out}_wf_max_raw_1460.png', dpi=200)


        plt.ylim(2400, 2750)
        plt.xlim(6500, 8500)

        plt.savefig(f'./plots/new_normScan/dataCleaning/N_minus_1/raw/except_{cut_out}_wf_max_raw_2615.png', dpi=200)
        plt.clf()
        plt.close()



        # ____________60 keV with fit________________________________________

        pgfenergy_hist, pgfebins, evars = pgh.get_hist(df['trapEftp_cal'], bins=50, range=[54, 65]) #range=[54, 65]
        plt.plot(pgfebins[1:], pgfenergy_hist, ds='steps', c='b', lw=1)
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
        suptitle = f'All cuts except {cut_out}\n{percent_surviving:.2f}% surviving cuts'

        plt.plot(pgfebins[1:], cage_utils.gauss_fit_func(pgfebins[1:], *e_pars), c = 'r')
        plt.plot(pgfebins[1:], pgfenergy_hist, ds='steps', c='b', lw=1)

        plt.xlabel('Energy (keV)', fontsize=16)
        plt.ylabel('counts', fontsize=16)

        plt.title('60 keV peak with gaussian fit', fontsize = 16)

        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)

        ax.text(0.95, 0.83, f'r = {radius} mm \ntheta = {angle_det} deg \nruntime {rt_min:.2f} \nmean: {mean_fit.2f} \nsigma: {width_fit:.3f} \nFWHM at 60 keV: {fwhm:.2f} \n{(fwhm/mean_fit)*100:.2f}%', position = (0.1, 0.85), transform=ax.transAxes, color='black', fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

        plt.savefig(f'./plots/new_normScan/dataCleaning/N_minus_1/raw/except_{cut_out}_fit_60keV_raw.png', dpi=200)
        plt.clf()
        plt.close()




if __name__=="__main__":
    main()
