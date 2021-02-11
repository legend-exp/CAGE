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
# plt.style.use('../clint.mpl')
from matplotlib.colors import LogNorm

# import boost_histogram as bh
# import pickle as pl

from pygama import DataGroup
import pygama.lh5 as lh5
import pygama.analysis.histograms as pgh
import pygama.analysis.peak_fitting as pgf

mpl.use('Agg')

def main():
    runs = [130, 133, 136]
    plot_spectra(runs, show=False, save=True)

def get_hists(runs, user=False, hit=True, cal=True, etype='trapEftp', bl_cut=True):

    hist_arr = []
    if cal==True:
            etype_cal = etype+'_cal'

    for run in runs:
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
            radius = int(radius)
            angle_det = int((-1*angle) - 90)
            if rotary <0:
                angle_det = int(angle + 270)
            print(f'Radius: {radius}; Angle: {angle_det}')

        else:
            radius = 'n/a'
            angle = 'n/a'
            angle_det = 'n/a'


        # print(etype, etype_cal, run)
        # exit()



        # get data and load into df
        lh5_dir = dg.lh5_user_dir if user else dg.lh5_dir

        if hit==True:
            print('Using hit files')
            file_list = lh5_dir + dg.fileDB['hit_path'] + '/' + dg.fileDB['hit_file']
            if run<117 and cal==True:
                df = lh5.load_dfs(file_list, ['energy', 'trapEmax', 'trapEmax_cal', 'bl','bl_sig','A_10','AoE', 'ts_sec', 'dcr_raw', 'dcr_ftp', 'dcr_max', 'tp_0','tp_10', 'tp_90', 'tp_50', 'tp_80', 'tp_max'], 'ORSIS3302DecoderForEnergy/hit')
            elif run>=117 and cal==True:
                df = lh5.load_dfs(file_list, ['energy', 'trapEmax', 'trapEftp', 'trapEmax_cal', 'trapEftp_cal', 'bl','bl_sig', 'bl_slope', 'lf_max', 'A_10','AoE', 'dcr', 'tp_0', 'tp_10', 'tp_90', 'tp_50', 'tp_80', 'tp_max'], 'ORSIS3302DecoderForEnergy/hit')

            elif run<117 and cal==False:
                df = lh5.load_dfs(file_list, [f'{etype}', 'bl','bl_sig','A_10','AoE', 'ts_sec', 'dcr_raw', 'dcr_ftp', 'dcr_max', 'tp_0', 'tp_10', 'tp_90', 'tp_50', 'tp_80', 'tp_max'], 'ORSIS3302DecoderForEnergy/hit')
            elif run>=117 and cal==False:
                df = lh5.load_dfs(file_list, [f'{etype}', 'bl','bl_sig', 'bl_slope', 'lf_max', 'A_10','AoE', 'dcr', 'tp_0', 'tp_10', 'tp_90', 'tp_50', 'tp_80', 'tp_max'], 'ORSIS3302DecoderForEnergy/hit')

        elif hit==False:
            print('Using dsp files')
            file_list = lh5_dir + dg.fileDB['dsp_path'] + '/' + dg.fileDB['dsp_file']
            if run<117 and cal==True:
                df = lh5.load_dfs(file_list, [f'{etype}', f'{etype_cal}', 'bl','bl_sig','A_10','AoE', 'ts_sec', 'dcr_raw', 'dcr_ftp', 'dcr_max', 'tp_0', 'tp_10', 'tp_90', 'tp_50', 'tp_80', 'tp_max'], 'ORSIS3302DecoderForEnergy/dsp')
            elif run>=117 and cal==True:
                df = lh5.load_dfs(file_list, [f'{etype}', f'{etype_cal}', 'bl','bl_sig', 'bl_slope', 'lf_max', 'A_10','AoE', 'dcr', 'tp_0', 'tp_10', 'tp_90','tp_50', 'tp_80', 'tp_max'], 'ORSIS3302DecoderForEnergy/dsp')

            elif run<117 and cal==False:
                df = lh5.load_dfs(file_list, [f'{etype}', 'bl','bl_sig','A_10','AoE', 'ts_sec', 'dcr_raw', 'dcr_ftp', 'dcr_max', 'tp_0', 'tp_10', 'tp_90', 'tp_50', 'tp_80', 'tp_max'], 'ORSIS3302DecoderForEnergy/dsp')
            elif run>=117 and cal==False:
                df = lh5.load_dfs(file_list, [f'{etype}', 'bl','bl_sig', 'bl_slope', 'lf_max', 'A_10','AoE', 'dcr', 'tp_0', 'tp_10', 'tp_90', 'tp_50', 'tp_80', 'tp_max'], 'ORSIS3302DecoderForEnergy/dsp')

        else:
            print('dont know what to do here! need to specify if working with calibrated/uncalibrated data, or dsp/hit files')

        if bl_cut==True:
            # use baseline cut
            print('Using baseline cut')
            if run <117:
                bl_cut_lo, bl_cut_hi = 8500, 10000
            if run>=117:
                bl_cut_lo, bl_cut_hi = 9700, 9760

            df_cut = df.query(f'bl > {bl_cut_lo} and bl < {bl_cut_hi}').copy()

        else:
            print('Not using baseline cut')
            df_cut = df

        # select energy type and energy range
        if cal==False:
            elo, ehi, epb = 0, 10000, 10 #entire enerty range trapEftp
            e_unit = ' (uncal)'
        elif cal==True:
            elo, ehi, epb = 0, 6000, 5
            etype=etype_cal
            e_unit = ' (keV)'

        # create energy histograms
        ene_hist, bins = np.histogram(df_cut[etype], bins=nbx, range=([elo, ehi]))
        ene_hist_norm = np.divide(ene_hist, (rt_min))

        hist_arr.append(ene_hist_norm)


    return(hist_arr, bins)

def plot_spectra(runs, show=False, save=True):

    hists, bins = get_hists(runs, user=False, hit=True, cal=True, etype='trapEftp', bl_cut=True)
    fig, ax = plt.subplots()
    fig.suptitle(f'Energy', horizontalalignment='center', fontsize=16)
    cmap = plt.cm.get_cmap('jet', len(hists))

    for hist, run, i in zip(hists, runs, range(len(hists))):
        plt.semilogy(bins[1:], hist, ds='steps', c=cmap(i), lw=1,
            label=f'Run {run}')

    plt.xlabel('Energy (keV)', fontsize=16)
    plt.ylabel('cts/min', fontsize=16)
    plt.setp(ax.get_xticklabels(), fontsize=14)
    plt.setp(ax.get_yticklabels(), fontsize=14)
    plt.legend()

    if show==True:
        plt.show()
    if save==True:
        print('Saving figure')
        plt.savefig('./plots/multi_spectra.png', dpi=200)


if __name__=="__main__":
    main()
