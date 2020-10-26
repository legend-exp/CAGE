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
# plt.style.use('../clint.mpl')
from matplotlib.colors import LogNorm

import boost_histogram as bh
import pickle as pl

from pygama import DataGroup
import pygama.io.lh5 as lh5
import pygama.analysis.histograms as pgh
import pygama.analysis.peak_fitting as pgf

def main():
    runs = [60, 42, 64, 44, 66, 48, 70, 50, 72, 54]

    plot_energy(runs)

def plot_energy(runs):
    radius_arr = []
    mean_energy_arr = []
    counts_arr = []


    for run in runs:
        # get run files
        dg = DataGroup('cage.json', load=True)
        str_query = f'run=={run} and skip==False'
        dg.file_keys.query(str_query, inplace=True)

        #get runtime, startime, runtype
        runtype_list = np.array(dg.file_keys['runtype'])
        runtype = runtype_list[0]
        rt_min = dg.file_keys['runtime'].sum()
        u_start = dg.file_keys.iloc[0]['startTime']
        t_start = pd.to_datetime(u_start, unit='s')

        # get scan position

        if runtype == 'alp':
            alphaDB = pd.read_hdf('alphaDB.h5')
            scan_pos = alphaDB.loc[alphaDB['run']==run]
            radius = np.array(scan_pos['radius'])[0]
            angle = np.array(scan_pos['angle'])[0]
            print(f'Radius: {radius}; Angle: {angle}')
        else:
            radius = 'bkg'
            angle = 'bkg'

        # get hit df
        lh5_dir = dg.lh5_user_dir #if user else dg.lh5_dir
        hit_list = lh5_dir + dg.file_keys['hit_path'] + '/' + dg.file_keys['hit_file']
        df_hit = lh5.load_dfs(hit_list, ['trapEmax', 'bl','bl_sig','A_10','AoE', 'ts_sec', 'dcr_raw', 'dcr_ftp', 'dcr_max', 'tp_10', 'tp_90', 'tp_50', 'tp_80', 'tp_max'], 'ORSIS3302DecoderForEnergy/hit')

        # use baseline cut
        df_cut = df_hit.query('bl > 8500 and bl < 10000').copy()

        #creat new DCR
        const = 0.0555
        df_cut['dcr_linoff'] = df_cut['dcr_raw'] + const*df_cut['trapEmax']

        # create cut for alphas
        alpha_cut = 'trapEmax> 2000 and trapEmax < 12000 and dcr_linoff > 30 and dcr_linoff < 200 and AoE > 0.03 and AoE < 0.06'
        new_dcr_cut = df_cut.query(alpha_cut).copy()
        # len(new_dcr_cut)

        alpha_energy = np.array(new_dcr_cut['trapEmax'])
        mean_energy = np.mean(alpha_energy)

        radius_arr.append(radius)
        mean_energy_arr.append(mean_energy)
        count_arr.append(len(mean_energy))

    energy_plot = plt.plot(radius_arr, mean_energy_arr, '.r')
    plt.savefig('./plots/normScan/energy_deg.png', dpi=200)

    rate_plot = plt.plot(radius_arr, count_arr, '.r')
    plt.savefig('./plots/normScan/counts_alpha.png', dpi=200)




if __name__=="__main__":
    main()
