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

import boost_histogram as bh
import pickle as pl

from pygama import DataGroup
import pygama.io.lh5 as lh5
import pygama.analysis.histograms as pgh
import pygama.analysis.peak_fitting as pgf

mpl.use('Agg')

def main():
    # runs = [60, 42, 64, 44, 66, 48, 70, 50, 72, 54]
    runs = [66, 68, 82, 84]

    timepoints(runs)


def timepoints(runs):

    for run in runs:
        # get run files
        dg = DataGroup('cage.json', load=True)
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
            alphaDB = pd.read_hdf('alphaDB.h5')
            scan_pos = alphaDB.loc[alphaDB['run']==run]
            radius = np.array(scan_pos['radius'])[0]
            angle = np.array(scan_pos['angle'])[0]
            angle_det = 270 + angle
            print(f'Radius: {radius}; Angle: {angle}')

        else:
            radius = 'n/a'
            angle = 'n/a'
            angle_det = 'n/a'

        # get hit df
        lh5_dir = dg.lh5_user_dir #if user else dg.lh5_dir
        hit_list = lh5_dir + dg.fileDB['hit_path'] + '/' + dg.fileDB['hit_file']
        df_hit = lh5.load_dfs(hit_list, ['trapEmax', 'trapEmax_cal', 'bl','bl_sig','A_10','AoE', 'ts_sec', 'dcr_raw', 'dcr_ftp', 'dcr_max', 'tp_0', 'tp_10', 'tp_50', 'tp_90', 'tp_max'], 'ORSIS3302DecoderForEnergy/hit')

        # use baseline cut
        df_cut = df_hit.query('bl > 8500 and bl < 10000').copy()

        #creat new DCR
        const = 0.0555
        df_cut['dcr_linoff'] = df_cut['dcr_raw'] + const*df_cut['trapEmax']

        #create 0-50
        df_cut['tp0_50'] = df_cut['tp_50']- df_cut['tp_0']

        #create 10-90
        df_cut['10-90'] = df_cut['tp_90']- df_cut['tp_10']

        #create 50-100
        df_cut['50-100'] = df_cut['tp_max']- df_cut['tp_50']

        #-------------------------------------
        # Plots before alpha cuts
        #--------------------

        # DCR vs tp_50___________

        fig, ax = plt.subplots()
        fig.suptitle(f'DCR vs 50% rise time', horizontalalignment='center', fontsize=16)

        dlo, dhi, dpb = -100, 200, 0.6
        tlo, thi, tpb = 0, 700, 10

        nbx = int((dhi-dlo)/dpb)
        nby = int((thi-tlo)/tpb)

        alpha_dcr_hist = plt.hist2d(df_cut['dcr_linoff'], df_cut['tp0_50'], bins=[nbx,nby],
                range=[[dlo, dhi], [tlo, thi]], cmap='viridis', norm=LogNorm())

        cb = plt.colorbar()
        cb.set_label("counts", ha = 'right', va='center', rotation=270, fontsize=14)
        cb.ax.tick_params(labelsize=12)
        ax.set_xlabel('DCR (arb)', fontsize=16)
        ax.set_ylabel('tp 0-50 (ns)', fontsize=16)
        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)

        # plt.legend()
        ax.text(0.95, 0.83, f'r = {radius} mm \ntheta = {angle_det} deg', verticalalignment='bottom',
                    horizontalalignment='right', transform=ax.transAxes, color='green', fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.95, 'pad': 10})

        plt.title(f'\n{runtype} run {run}, {rt_min:.2f} mins', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'./plots/normScan/cal_normScan/{runtype}_dcr_vs_tp0_50_run{run}.png', dpi=200)
        # plt.show()
        plt.clf()
        plt.close()

        # DCR vs 10-90___________

        fig, ax = plt.subplots()
        fig.suptitle(f'DCR vs 10-90% rise time', horizontalalignment='center', fontsize=16)

        dlo, dhi, dpb = -100, 200, 0.6
        tlo, thi, tpb = 0, 600, 10

        nbx = int((dhi-dlo)/dpb)
        nby = int((thi-tlo)/tpb)

        alpha_dcr_hist = plt.hist2d(df_cut['dcr_linoff'], df_cut['10-90'], bins=[nbx,nby],
                range=[[dlo, dhi], [tlo, thi]], cmap='viridis', norm=LogNorm())

        cb = plt.colorbar()
        cb.set_label("counts", ha = 'right', va='center', rotation=270, fontsize=14)
        cb.ax.tick_params(labelsize=12)
        ax.set_xlabel('DCR (arb)', fontsize=16)
        ax.set_ylabel('tp 10-90 (ns)', fontsize=16)
        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)

        # plt.legend()
        ax.text(0.95, 0.83, f'r = {radius} mm \ntheta = {angle_det} deg', verticalalignment='bottom',
                    horizontalalignment='right', transform=ax.transAxes, color='green', fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.95, 'pad': 10})

        plt.title(f'\n{runtype} run {run}, {rt_min:.2f} mins', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'./plots/normScan/cal_normScan/{runtype}_dcr_vs_tp10_90_run{run}.png', dpi=200)
        # plt.show()
        plt.clf()
        plt.close()

        # DCR vs 50-100___________

        fig, ax = plt.subplots()
        fig.suptitle(f'DCR vs 50-100% rise time', horizontalalignment='center', fontsize=16)

        dlo, dhi, dpb = -100, 200, 0.6
        tlo, thi, tpb = 0, 1000, 10

        nbx = int((dhi-dlo)/dpb)
        nby = int((thi-tlo)/tpb)

        alpha_dcr_hist = plt.hist2d(df_cut['dcr_linoff'], df_cut['50-100'], bins=[nbx,nby],
                range=[[dlo, dhi], [tlo, thi]], cmap='viridis', norm=LogNorm())

        cb = plt.colorbar()
        cb.set_label("counts", ha = 'right', va='center', rotation=270, fontsize=14)
        cb.ax.tick_params(labelsize=12)
        ax.set_xlabel('DCR (arb)', fontsize=16)
        ax.set_ylabel('tp 10-90 (ns)', fontsize=16)
        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)

        # plt.legend()
        ax.text(0.95, 0.83, f'r = {radius} mm \ntheta = {angle_det} deg', verticalalignment='bottom',
                    horizontalalignment='right', transform=ax.transAxes, color='green', fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.95, 'pad': 10})

        plt.title(f'\n{runtype} run {run}, {rt_min:.2f} mins', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'./plots/normScan/cal_normScan/{runtype}_dcr_vs_tp50_100_run{run}.png', dpi=200)
        # plt.show()
        plt.clf()
        plt.close()

if __name__=="__main__":
    main()
