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
#     runs = [60, 42, 64, 44, 66, 48, 70, 50, 72, 54]
    runs = [87]

#     plot_energy(runs)
    dcr_AvE(runs, cut=False)

def dcr_AvE(runs, cut=True):

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
        df_hit = lh5.load_dfs(hit_list, ['trapEmax', 'trapEmax_cal', 'bl','bl_sig','A_10','AoE', 'ts_sec', 'dcr_raw', 'dcr_ftp', 'dcr_max', 'tp_0', 'tp_10', 'tp_90', 'tp_50', 'tp_80', 'tp_max'], 'ORSIS3302DecoderForEnergy/hit')

        # use baseline cut
        df_cut = df_hit.query('bl > 8500 and bl < 10000').copy()

        #creat new DCR
        const = 0.0555
        
        if run>86:
            const = -0.0225
        df_cut['dcr_linoff'] = df_cut['dcr_raw'] + const*df_cut['trapEmax']

        #create 0-50
        df_cut['tp0_50'] = df_cut['tp_50']- df_cut['tp_0']

        # create cut for alphas
        alpha_cut = 'dcr_linoff > 25 and dcr_linoff < 200 and tp0_50 > 100 and tp0_50 < 400 and trapEmax_cal < 6000'
        new_dcr_cut = df_cut.query(alpha_cut).copy()
        # len(new_dcr_cut)

        #-------------------------------------
        # Plots before alpha cuts
        #--------------------

        # Make calibrated energy spectrum_________

        fig, ax = plt.subplots()
        fig.suptitle(f'Energy', horizontalalignment='center', fontsize=16)
        elo, ehi, epb = 0, 6000, 10
        # elo, ehi, epb = 0, 3000, 10
        # elo, ehi, epb = 0, 6000, 10


        nbx = int((ehi-elo)/epb)

        energy_hist, bins = np.histogram(df_cut['trapEmax_cal'], bins=nbx,
                range=[elo, ehi])
        energy_rt = np.divide(energy_hist, rt_min * 60)

        plt.semilogy(bins[1:], energy_rt, ds='steps', c='b', lw=1) #, label=f'{etype}'

        ax.set_xlabel('Energy (keV)', fontsize=16)
        ax.set_ylabel('counts/sec', fontsize=16)
        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)

        ax.text(0.95, 0.83, f'r = {radius} mm \ntheta = {angle_det} deg', verticalalignment='bottom',
                    horizontalalignment='right', transform=ax.transAxes, color='green', fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

        # plt.legend()
        plt.title(f'\n{runtype} run {run}, {rt_min:.2f} mins', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'./plots/normScan/cal_normScan/{runtype}_energy_run{run}.png', dpi=200)
        plt.clf()
        plt.close()


        # AoE vs E---------------------------------
        fig, ax = plt.subplots()
        alo, ahi, apb = 0.0, 0.09, 0.0001
        if run>=60:
            alo, ahi, apb = 0.005, 0.0905, 0.0001
        elo, ehi, epb = 0, 6000, 10
        # elo, ehi, epb = 0, 3000, 10
        # elo, ehi, epb = 0, 6000, 10


        nbx = int((ehi-elo)/epb)
        nby = int((ahi-alo)/apb)

        fig.suptitle(f'A/E vs Energy', horizontalalignment='center', fontsize=16)

        h = plt.hist2d(df_cut['trapEmax_cal'], df_cut['AoE'], bins=[nbx,nby],
                    range=[[elo, ehi], [alo, ahi]], cmap='viridis', norm=LogNorm())

        cb = plt.colorbar()
        cb.set_label("counts", ha = 'right', va='center', rotation=270, fontsize=14)
        cb.ax.tick_params(labelsize=12)
        ax.set_xlabel('Energy (keV)', fontsize=16)
        ax.set_ylabel('A/E (arb)', fontsize=16)
        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)


        ax.text(0.95, 0.83, f'r = {radius} mm \ntheta = {angle_det} deg', verticalalignment='bottom',
                    horizontalalignment='right', transform=ax.transAxes, color='green', fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

        # plt.legend()
        plt.title(f'\n{runtype} run {run}, {rt_min:.2f} mins', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'./plots/normScan/cal_normScan/{runtype}_AoE_run{run}.png', dpi=200)
        # plt.show()

        plt.clf()
        plt.close()

        # DCR vs E___________

        fig, ax = plt.subplots()
        dlo, dhi, dpb = -100, 300, 0.6
        elo, ehi, epb = 0, 6000, 10
        # elo, ehi, epb = 0, 3000, 10
        # elo, ehi, epb = 0, 6000, 10

        nbx = int((ehi-elo)/epb)
        nby = int((dhi-dlo)/dpb)

        fig.suptitle(f'DCR vs Energy', horizontalalignment='center', fontsize=16)

        h = plt.hist2d(df_cut['trapEmax_cal'], df_cut['dcr_linoff'], bins=[nbx,nby],
                    range=[[elo, ehi], [dlo, dhi]], cmap='viridis', norm=LogNorm())

        cb = plt.colorbar()
        cb.set_label("counts", ha = 'right', va='center', rotation=270, fontsize=14)
        cb.ax.tick_params(labelsize=12)
        ax.set_xlabel('Energy (keV)', fontsize=16)
        ax.set_ylabel('DCR (arb)', fontsize=16)
        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)

        # plt.legend()

        ax.text(0.95, 0.83, f'r = {radius} mm \ntheta = {angle_det} deg', verticalalignment='bottom',
                    horizontalalignment='right', transform=ax.transAxes, color='green', fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

        plt.title(f'\n{runtype} run {run}, {rt_min:.2f} mins', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'./plots/normScan/cal_normScan/{runtype}_dcr_run{run}.png', dpi=200)
        # plt.show()
        plt.clf()
        plt.close()

        # DCR vs A/E___________

        fig, ax = plt.subplots()
        dlo, dhi, dpb = -100, 300, 0.6
        alo, ahi, apb = 0.0, 0.09, 0.0001
        if run>=60:
            alo, ahi, apb = 0.005, 0.0905, 0.0001

        nbx = int((ahi-alo)/apb)
        nby = int((dhi-dlo)/dpb)

        fig.suptitle(f'A/E vs DCR', horizontalalignment='center', fontsize=16)

        h = plt.hist2d(df_cut['AoE'], df_cut['dcr_linoff'], bins=[nbx,nby],
                    range=[[alo, ahi], [dlo, dhi]], cmap='viridis', norm=LogNorm())

        cb = plt.colorbar()
        cb.set_label("counts", ha = 'right', va='center', rotation=270, fontsize=14)
        cb.ax.tick_params(labelsize=12)
        ax.set_xlabel('A/E (arb)', fontsize=16)
        ax.set_ylabel('DCR (arb)', fontsize=16)
        plt.setp(ax.get_xticklabels(), fontsize=12)
        plt.setp(ax.get_yticklabels(), fontsize=14)

        # plt.legend()
        ax.text(0.95, 0.83, f'r = {radius} mm \ntheta = {angle_det} deg', verticalalignment='bottom',
                    horizontalalignment='right', transform=ax.transAxes, color='green', fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

        plt.title(f'\n{runtype} run {run}, {rt_min:.2f} mins', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'./plots/normScan/cal_normScan/{runtype}_AoE_vs_dcr_run{run}.png', dpi=200)
        # plt.show()
        plt.clf()
        plt.close()

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

        # 1D AoE_________

        fig, ax = plt.subplots()
        fig.suptitle(f'A/E', horizontalalignment='center', fontsize=16)

        alo, ahi, apb = 0.0, 0.09, 0.0001
        if run>=60:
            alo, ahi, apb = 0.005, 0.0905, 0.0001
        nbx = int((ahi-alo)/apb)

        aoe_hist, bins = np.histogram(df_cut['AoE'], bins=nbx,
                range=[alo, ahi])

        plt.semilogy(bins[1:], aoe_hist, ds='steps', c='b', lw=1) #, label=f'{etype}'

        ax.set_xlabel('A/E (arb)', fontsize=16)
        ax.set_ylabel('counts', fontsize=16)
        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)

        ax.text(0.95, 0.83, f'r = {radius} mm \ntheta = {angle_det} deg', verticalalignment='bottom',
                    horizontalalignment='right', transform=ax.transAxes, color='green', fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

        # plt.legend()
        plt.title(f'\n{runtype} run {run}, {rt_min:.2f} mins', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'./plots/normScan/cal_normScan/{runtype}_1d_aoe_run{run}.png', dpi=200)
        plt.clf()
        plt.close()
        

        #-------------------------------------
        # Plots after alpha cuts
        #--------------------
        
        if cut==False:
            exit()
        else:
            continue

        # Make calibrated energy spectrum_________

        fig, ax = plt.subplots()
        fig.suptitle(f'Energy after cut', horizontalalignment='center', fontsize=16)
        elo, ehi, epb = 0, 6000, 10
        # elo, ehi, epb = 0, 3000, 10
        # elo, ehi, epb = 0, 6000, 10


        nbx = int((ehi-elo)/epb)

        energy_hist, bins = np.histogram(new_dcr_cut['trapEmax_cal'], bins=nbx,
                range=[elo, ehi])
        energy_rt = np.divide(energy_hist, rt_min * 60)

        plt.semilogy(bins[1:], energy_rt, ds='steps', c='b', lw=1) #, label=f'{etype}'

        ax.set_xlabel('Energy (keV)', fontsize=16)
        ax.set_ylabel('counts/sec', fontsize=16)
        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)

        ax.text(0.95, 0.83, f'r = {radius} mm \ntheta = {angle_det} deg', verticalalignment='bottom',
                    horizontalalignment='right', transform=ax.transAxes, color='green', fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

        # plt.legend()
        plt.title(f'\n{runtype} run {run}, {rt_min:.2f} mins', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'./plots/normScan/cal_normScan/{runtype}_cut_energy_run{run}.png', dpi=200)
        plt.clf()
        plt.close()

        # AoE vs E---------------------------------
        fig, ax = plt.subplots()
        alo, ahi, apb = 0.0, 0.09, 0.0001
        if run>=60:
            alo, ahi, apb = 0.005, 0.0905, 0.0001
        elo, ehi, epb = 0, 6000, 10
        # elo, ehi, epb = 0, 3000, 10
        # elo, ehi, epb = 0, 6000, 10


        nbx = int((ehi-elo)/epb)
        nby = int((ahi-alo)/apb)

        fig.suptitle(f'A/E vs Energy after cut', horizontalalignment='center', fontsize=16)

        h = plt.hist2d(new_dcr_cut['trapEmax_cal'], new_dcr_cut['AoE'], bins=[nbx,nby],
                    range=[[elo, ehi], [alo, ahi]], cmap='viridis', norm=LogNorm())

        cb = plt.colorbar()
        cb.set_label("counts", ha = 'right', va='center', rotation=270, fontsize=14)
        cb.ax.tick_params(labelsize=12)
        ax.set_xlabel('Energy (keV)', fontsize=16)
        ax.set_ylabel('A/E (arb)', fontsize=16)
        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)

        ax.text(0.95, 0.83, f'r = {radius} mm \ntheta = {angle_det} deg', verticalalignment='bottom',
                    horizontalalignment='right', transform=ax.transAxes, color='green', fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

        # plt.legend()
        plt.title(f'\n{runtype} run {run}, {rt_min:.2f} mins', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'./plots/normScan/cal_normScan/{runtype}_cut_AoE_run{run}.png', dpi=200)
        # plt.show()

        plt.clf()
        plt.close()

        # DCR vs E___________

        fig, ax = plt.subplots()
        dlo, dhi, dpb = -100, 300, 0.6
        elo, ehi, epb = 0, 6000, 10
        # elo, ehi, epb = 0, 3000, 10
        # elo, ehi, epb = 0, 6000, 10

        nbx = int((ehi-elo)/epb)
        nby = int((dhi-dlo)/dpb)

        fig.suptitle(f'DCR vs Energy after cut', horizontalalignment='center', fontsize=16)

        h = plt.hist2d(new_dcr_cut['trapEmax_cal'], new_dcr_cut['dcr_linoff'], bins=[nbx,nby],
                    range=[[elo, ehi], [dlo, dhi]], cmap='viridis', norm=LogNorm())

        cb = plt.colorbar()
        cb.set_label("counts", ha = 'right', va='center', rotation=270, fontsize=14)
        cb.ax.tick_params(labelsize=12)
        ax.set_xlabel('Energy (keV)', fontsize=16)
        ax.set_ylabel('DCR (arb)', fontsize=16)
        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)

        # plt.legend()

        ax.text(0.95, 0.83, f'r = {radius} mm \ntheta = {angle_det} deg', verticalalignment='bottom',
                    horizontalalignment='right', transform=ax.transAxes, color='green', fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

        plt.title(f'\n{runtype} run {run}, {rt_min:.2f} mins', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'./plots/normScan/cal_normScan/{runtype}_cut_dcr_run{run}.png', dpi=200)
        # plt.show()
        plt.clf()
        plt.close()

        # DCR vs A/E___________

        fig, ax = plt.subplots()
        dlo, dhi, dpb = -100, 300, 0.6
        alo, ahi, apb = 0.0, 0.09, 0.0001
        if run>=60:
            alo, ahi, apb = 0.005, 0.0905, 0.0001

        nbx = int((ahi-alo)/apb)
        nby = int((dhi-dlo)/dpb)

        fig.suptitle(f'A/E vs DCR after cut', horizontalalignment='center', fontsize=16)

        h = plt.hist2d(new_dcr_cut['AoE'], new_dcr_cut['dcr_linoff'], bins=[nbx,nby],
                    range=[[alo, ahi], [dlo, dhi]], cmap='viridis', norm=LogNorm())

        cb = plt.colorbar()
        cb.set_label("counts", ha = 'right', va='center', rotation=270, fontsize=14)
        cb.ax.tick_params(labelsize=12)
        ax.set_xlabel('A/E (arb)', fontsize=16)
        ax.set_ylabel('DCR (arb)', fontsize=16)
        plt.setp(ax.get_xticklabels(), fontsize=12)
        plt.setp(ax.get_yticklabels(), fontsize=14)
        # plt.legend()
        ax.text(0.95, 0.83, f'r = {radius} mm \ntheta = {angle_det} deg', verticalalignment='bottom',
                    horizontalalignment='right', transform=ax.transAxes, color='green', fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

        plt.title(f'\n{runtype} run {run}, {rt_min:.2f} mins', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'./plots/normScan/cal_normScan/{runtype}_cut_AoE_vs_dcr_run{run}.png', dpi=200)
        # plt.show()
        plt.clf()
        plt.close()

        # DCR vs tp_50___________

        fig, ax = plt.subplots()
        fig.suptitle(f'DCR vs 50% rise time after cut', horizontalalignment='center', fontsize=16)

        dlo, dhi, dpb = -100, 200, 0.6
        tlo, thi, tpb = 0, 700, 10

        nbx = int((dhi-dlo)/dpb)
        nby = int((thi-tlo)/tpb)

        alpha_dcr_hist = plt.hist2d(new_dcr_cut['dcr_linoff'], new_dcr_cut['tp0_50'], bins=[nbx,nby],
                range=[[dlo, dhi], [tlo, thi]], cmap='viridis', norm=LogNorm())

        cb = plt.colorbar()
        cb.set_label("counts", ha = 'right', va='center', rotation=270, fontsize=14)
        cb.ax.tick_params(labelsize=12)
        ax.set_xlabel('DCR (arb)', fontsize=16)
        ax.set_ylabel('tp 0-50 (ns)', fontsize=16)
        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)

        # plt.legend()
        ax.text(0.95, 0.83, f'r = {radius} mm \n theta = {angle_det} deg', verticalalignment='bottom',
                    horizontalalignment='right', transform=ax.transAxes, color='green', fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

        plt.title(f'\n{runtype} run {run}, {rt_min:.2f} mins', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'./plots/normScan/cal_normScan/{runtype}_cut_dcr_vs_tp0_50_run{run}.png', dpi=200)
        # plt.show()
        plt.clf()
        plt.close()

        # 1D AoE_________

        fig, ax = plt.subplots()
        fig.suptitle(f'A/E after cut', horizontalalignment='center', fontsize=16)
        alo, ahi, apb = 0.0, 0.09, 0.0001
        if run>=60:
            alo, ahi, apb = 0.005, 0.0905, 0.0001
        nbx = int((ahi-alo)/apb)

        aoe_hist, bins = np.histogram(new_dcr_cut['AoE'], bins=nbx,
                range=[alo, ahi])

        plt.semilogy(bins[1:], aoe_hist, ds='steps', c='b', lw=1) #, label=f'{etype}'

        ax.set_xlabel('A/E (arb)', fontsize=16)
        ax.set_ylabel('counts', fontsize=16)
        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)

        ax.text(0.95, 0.83, f'r = {radius} mm \ntheta = {angle_det} deg', verticalalignment='bottom',
                    horizontalalignment='right', transform=ax.transAxes, color='green', fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

        # plt.legend()
        plt.title(f'\n{runtype} run {run}, {rt_min:.2f} mins', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'./plots/normScan/cal_normScan/{runtype}_cut__1d_aoe_run{run}.png', dpi=200)
        plt.clf()
        plt.close()
        
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
        df_hit = lh5.load_dfs(hit_list, ['trapEmax', 'trapEmax_cal', 'bl','bl_sig','A_10','AoE', 'ts_sec', 'dcr_raw', 'dcr_ftp', 'dcr_max', 'tp_0', 'tp_10', 'tp_90', 'tp_50', 'tp_80', 'tp_max'], 'ORSIS3302DecoderForEnergy/hit')

        # use baseline cut
        df_cut = df_hit.query('bl > 8500 and bl < 10000').copy()

        #creat new DCR
        const = 0.0555
        df_cut['dcr_linoff'] = df_cut['dcr_raw'] + const*df_cut['trapEmax']

        #create 0-50
        df_cut['tp0_50'] = df_cut['tp_50']- df_cut['tp_0']

        # create cut for alphas
        alpha_cut = 'dcr_linoff > 25 and dcr_linoff < 200 and tp0_50 > 100 and tp0_50 < 400 and trapEmax_cal < 6000'
        new_dcr_cut = df_cut.query(alpha_cut).copy()
        # len(new_dcr_cut)

        alpha_energy = np.array(new_dcr_cut['trapEmax_cal'])
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
    fig, ax = plt.subplots()

    energy_plot = plt.errorbar(radius_arr_1, mean_energy_arr_1, yerr=std_energy_arr_1, marker = '.', ls='none', color = 'red', label='Scan 1')
    ax.set_xlabel('Radial position (mm)', fontsize=16)
    ax.set_ylabel('Mean energy (keV)', fontsize=16)
    plt.setp(ax.get_xticklabels(), fontsize=14)
    plt.setp(ax.get_yticklabels(), fontsize=14)


#     plt.yscale('log')
    plt.title('Mean energy of alphas by radial position \nnormal incidence', fontsize=16)


    plt.errorbar(radius_arr_2, mean_energy_arr_2, yerr=std_energy_arr_2, marker = '.', ls='none', color = 'blue', label='Scan 2')
    plt.legend()
    plt.tight_layout()

    plt.savefig('./plots/normScan/cal_normScan/errorbars_energy_deg.png', dpi=200)

    plt.clf()
    plt.close()

    fig, ax = plt.subplots()
    dcr_plot = plt.errorbar(radius_arr_1, mean_dcr_arr_1, yerr=std_dcr_arr_1, marker = '.', ls='none', color = 'red', label='Scan 1')
    ax.set_xlabel('Radial position (mm)', fontsize=16)
    ax.set_ylabel('Mean DCR value (arb)', fontsize=16)
    plt.setp(ax.get_xticklabels(), fontsize=14)
    plt.setp(ax.get_yticklabels(), fontsize=14)

    #    plt.yscale('log')
    plt.title('Mean DCR value by radial position \nnormal incidence', fontsize=16)


    plt.errorbar(radius_arr_2, mean_dcr_arr_2, yerr=std_dcr_arr_2, marker = '.', ls='none', color = 'blue', label='Scan 2')
    plt.legend()
    plt.tight_layout()

    plt.savefig('./plots/normScan/cal_normScan/errorbars_dcr_avg.png', dpi=200)

    plt.clf()
    plt.close()

    # make plots without errorbars
    fig, ax = plt.subplots()
    energy_plot = plt.plot(radius_arr_1, mean_energy_arr_1, '.r', label='Scan 1')
    ax.set_xlabel('Radial position (mm)', fontsize=16)
    ax.set_ylabel('Mean energy (keV)', fontsize=16)
    plt.setp(ax.get_xticklabels(), fontsize=14)
    plt.setp(ax.get_yticklabels(), fontsize=14)


#     plt.yscale('log')
    plt.title('Mean energy of alphas by radial position \nnormal incidence', fontsize=16)


    plt.plot(radius_arr_2, mean_energy_arr_2, '.b', label='Scan 2')
    plt.legend()
    plt.tight_layout()

    plt.savefig('./plots/normScan/cal_normScan/energy_deg.png', dpi=200)

    plt.clf()
    plt.close()

    fig, ax = plt.subplots()
    dcr_plot = plt.plot(radius_arr_1, mean_dcr_arr_1, '.r', label='Scan 1')
    ax.set_xlabel('Radial position (mm)', fontsize=16)
    ax.set_ylabel('Mean DCR value (arb)', fontsize=16)
    plt.setp(ax.get_xticklabels(), fontsize=14)
    plt.setp(ax.get_yticklabels(), fontsize=14)

    #    plt.yscale('log')
    plt.title('Mean DCR value by radial position \nnormal incidence', fontsize=16)

    plt.plot(radius_arr_2, mean_dcr_arr_2, '.b', label='Scan 2')
    plt.legend()
    plt.tight_layout()

    plt.savefig('./plots/normScan/cal_normScan/dcr_avg.png', dpi=200)

    # plt.clf()
    plt.close()

#     rate_plot = plt.plot(radius_arr, count_arr, '.r')
#     plt.xlabel('Radial position (mm)')
#     plt.ylabel('Total counts)')
# #     plt.yscale('log')
#     plt.title('Alpha counts by radial position (based on DCR cut)')
#     plt.savefig('./plots/normScan/counts_alpha.png', dpi=200)
#     print(len(count_arr), len(radius_arr))




if __name__=="__main__":
    main()
