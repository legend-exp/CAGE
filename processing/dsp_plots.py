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
    radius_arr_1 = []
    mean_energy_arr_1 = []
    std_energy_arr_1 = []
    mean_dcr_arr_1 = []
    std_dcr_arr_1 = []
    count_arr_1 = []

    radius_arr_2 = []
    mean_energy_arr_2 = []
    std_energy_arr_2 = []
    mean_dcr_arr_2 = []
    std_dcr_arr_2 = []
    count_arr_2 = []


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
            angle_det = 270 + angle
            print(f'Radius: {radius}; Angle: {angle}')

        else:
            radius = 'n/a'
            angle = 'n/a'
            angle_det = 'n/a'

        # get hit df
        lh5_dir = dg.lh5_user_dir #if user else dg.lh5_dir
        hit_list = lh5_dir + dg.file_keys['hit_path'] + '/' + dg.file_keys['hit_file']
        df_hit = lh5.load_dfs(hit_list, ['trapEmax', 'bl','bl_sig','A_10','AoE', 'ts_sec', 'dcr_raw', 'dcr_ftp', 'dcr_max', 'tp_0', 'tp_10', 'tp_90', 'tp_50', 'tp_80', 'tp_max'], 'ORSIS3302DecoderForEnergy/hit')

        # use baseline cut
        df_cut = df_hit.query('bl > 8500 and bl < 10000').copy()

        #creat new DCR
        const = 0.0555
        df_cut['dcr_linoff'] = df_cut['dcr_raw'] + const*df_cut['trapEmax']

        #create 0-50
        df_cut['tp0_50'] = df_cut['tp_50']- df_cut['tp_0']

        # create cut for alphas
        alpha_cut = 'dcr_linoff > 25 and dcr_linoff < 200 and tp0_50 > 100 and tp0_50 < 400 and trapEmax < 12000'
        new_dcr_cut = df_cut.query(alpha_cut).copy()
        # len(new_dcr_cut)

        alpha_energy = np.array(new_dcr_cut['trapEmax'])
        mean_energy = np.mean(alpha_energy)
        std_energy = np.std(alpha_energy)
#         std_energy = np.sqrt(len(new_dcr_cut['trapEmax']))

        alpha_dcr = np.array(new_dcr_cut['dcr_linoff'])
        mean_dcr = np.mean(alpha_dcr)
        std_dcr = np.std(alpha_dcr)
#         std_dcr = np.sqrt((len(new_dcr_cut['dcr_linoff'])))
        
        print(f'Energy std: {std_energy} \n DCR std: {std_dcr}')

        if radius%5 == 0:
            radius_arr_1.append(radius)
            mean_energy_arr_1.append(mean_energy)
            std_energy_arr_1.append(std_energy)
            mean_dcr_arr_1.append(mean_dcr)
            std_dcr_arr_1.append(std_dcr)
            count_arr_1.append(len(alpha_energy))

        else:
            radius_arr_2.append(radius)
            mean_energy_arr_2.append(mean_energy)
            std_energy_arr_2.append(std_energy)
            mean_dcr_arr_2.append(mean_dcr)
            std_dcr_arr_2.append(std_dcr)
            count_arr_2.append(len(alpha_energy))
            
    # make plots with errorbars

    energy_plot = plt.errorbar(radius_arr_1, mean_energy_arr_1, yerr=std_energy_arr_1, marker = '.', ls='none', color = 'red', label='Scan 1')
    plt.xlabel('Radial position (mm)')
    plt.ylabel('Mean energy (trapEmax; uncal)')
#     plt.yscale('log')
    plt.title('Mean energy of alphas by radial position; normal incidence')


    plt.errorbar(radius_arr_2, mean_energy_arr_2, yerr=std_energy_arr_2, marker = '.', ls='none', color = 'blue', label='Scan 2')
    plt.legend()

    plt.savefig('./plots/normScan/errorbars_energy_deg.png', dpi=200)

    plt.clf()

    dcr_plot = plt.errorbar(radius_arr_1, mean_dcr_arr_1, yerr=std_dcr_arr_1, marker = '.', ls='none', color = 'red', label='Scan 1')
    plt.xlabel('Radial position (mm)')
    plt.ylabel('Mean DCR value (arb)')
    #    plt.yscale('log')
    plt.title('Mean DCR value by radial position; normal incidence')


    plt.errorbar(radius_arr_2, mean_dcr_arr_2, yerr=std_dcr_arr_2, marker = '.', ls='none', color = 'blue', label='Scan 2')
    plt.legend()

    plt.savefig('./plots/normScan/errorbars_dcr_avg.png', dpi=200)
    
    plt.clf()
    
    # make plots without errorbars
    
    energy_plot = plt.plot(radius_arr_1, mean_energy_arr_1, '.r', label='Scan 1')
    plt.xlabel('Radial position (mm)')
    plt.ylabel('Mean energy (trapEmax; uncal)')
#     plt.yscale('log')
    plt.title('Mean energy of alphas by radial position; normal incidence')


    plt.plot(radius_arr_2, mean_energy_arr_2, '.b', label='Scan 2')
    plt.legend()

    plt.savefig('./plots/normScan/energy_deg.png', dpi=200)

    plt.clf()

    dcr_plot = plt.plot(radius_arr_1, mean_dcr_arr_1, '.r', label='Scan 1')
    plt.xlabel('Radial position (mm)')
    plt.ylabel('Mean DCR value (arb)')
    #    plt.yscale('log')
    plt.title('Mean DCR value by radial position; normal incidence')


    plt.plot(radius_arr_2, mean_dcr_arr_2, '.b', label='Scan 2')
    plt.legend()

    plt.savefig('./plots/normScan/dcr_avg.png', dpi=200)

    # plt.clf()

#     rate_plot = plt.plot(radius_arr, count_arr, '.r')
#     plt.xlabel('Radial position (mm)')
#     plt.ylabel('Total counts)')
# #     plt.yscale('log')
#     plt.title('Alpha counts by radial position (based on DCR cut)')
#     plt.savefig('./plots/normScan/counts_alpha.png', dpi=200)
#     print(len(count_arr), len(radius_arr))




if __name__=="__main__":
    main()
