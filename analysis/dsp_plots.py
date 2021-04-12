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
    runs = [38, 60, 42, 64, 44, 66, 48, 70, 50, 72, 54]
    # runs = [120, 121, 123, 124, 126, 128, 129, 131, 132, 134, 135, 137, 143]
#     runs = [60, 64, 66, 70, 72]
#     runs = [60, 38]
#     alp_runs = [137, 143]
#     bkg_runs = [136, 136]
    campaign = 'normScan/'

    user = True
    hit = True
    cal = True
    etype = 'trapEftp'

#     plot_energy(runs)
    # dcr_AvE(runs, user, hit, cal, etype, cut=False)
    normalized_dcr_AvE(runs, user, hit, cal, etype, cut=False, campaign=campaign)
#     bkg_sub_dcr_AvE(alp_runs, bkg_runs, user, hit, cal, etype, cut=False)

def bkg_sub_dcr_AvE(alp_runs, bkg_runs, user=False, hit=True, cal=True, etype='trapEmax', cut=True, campaign=''):

    if cal==True:
        etype_cal = etype+'_cal'

    elif cal==False:
        print('Do not recommend background-subtracting uncalibrated data in case there are gain shifts!')

    for run, bkg in zip(alp_runs, bkg_runs):
        print(f'run: {run}\n bkg_run: {bkg}')

        #setup bkg run files
        bkg_dg = DataGroup('$CAGE_SW/processing/cage.json', load=True)
        bkg_query = f'run=={bkg} and skip==False'
        bkg_dg.fileDB.query(bkg_query, inplace=True)

        #get bkg runtime, startime
        bkg_rt_min = bkg_dg.fileDB['runtime'].sum()
        bkg_u_start = bkg_dg.fileDB.iloc[0]['startTime']
        bkg_t_start = pd.to_datetime(bkg_u_start, unit='s')

        # setup alpha run files
        alp_dg = DataGroup('$CAGE_SW/processing/cage.json', load=True)
        alp_query = f'run=={run} and skip==False'
        alp_dg.fileDB.query(alp_query, inplace=True)

        #get alp runtime, startime, runtype
        alp_runtype_list = np.array(alp_dg.fileDB['runtype'])
        alp_runtype = alp_runtype_list[0]
        alp_rt_min = alp_dg.fileDB['runtime'].sum()
        alp_u_start = alp_dg.fileDB.iloc[0]['startTime']
        alp_t_start = pd.to_datetime(alp_u_start, unit='s')

        # get scan position

        # doing this in case we want to subtract two background runs in the future and use this function,
        # one will be labeled "alpha" but not actually and alpha run.
        if alp_runtype == 'alp':
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



        # get bkg and alp data and load into separate df's
        bkg_lh5_dir = bkg_dg.lh5_user_dir if user else bkg_dg.lh5_dir
        alp_lh5_dir = alp_dg.lh5_user_dir if user else alp_dg.lh5_dir

        if hit==True:
            print('Using hit files')
            bkg_file_list = bkg_lh5_dir + bkg_dg.fileDB['hit_path'] + '/' + bkg_dg.fileDB['hit_file']
            alp_file_list = alp_lh5_dir + alp_dg.fileDB['hit_path'] + '/' + alp_dg.fileDB['hit_file']
            if run<117 and cal==True:
                bkg_df = lh5.load_dfs(bkg_file_list, ['energy', 'trapEmax', 'trapEmax_cal', 'bl','bl_sig','A_10','AoE', 'ts_sec', 'dcr_raw', 'dcr_ftp', 'dcr_max', 'tp_0','tp_10', 'tp_90', 'tp_50', 'tp_80', 'tp_max'], 'ORSIS3302DecoderForEnergy/hit')
                alp_df = lh5.load_dfs(alp_file_list, ['energy', 'trapEmax', 'trapEmax_cal', 'bl','bl_sig','A_10','AoE', 'ts_sec', 'dcr_raw', 'dcr_ftp', 'dcr_max', 'tp_0','tp_10', 'tp_90', 'tp_50', 'tp_80', 'tp_max'], 'ORSIS3302DecoderForEnergy/hit')
            elif run>=117 and cal==True:
                bkg_df = lh5.load_dfs(bkg_file_list, ['energy', 'trapEmax', 'trapEftp', 'trapEmax_cal', 'trapEftp_cal', 'bl','bl_sig', 'bl_slope', 'lf_max', 'A_10','AoE', 'dcr', 'tp_0', 'tp_10', 'tp_90', 'tp_50', 'tp_80', 'tp_max'], 'ORSIS3302DecoderForEnergy/hit')
                alp_df = lh5.load_dfs(alp_file_list, ['energy', 'trapEmax', 'trapEftp', 'trapEmax_cal', 'trapEftp_cal', 'bl','bl_sig', 'bl_slope', 'lf_max', 'A_10','AoE', 'dcr', 'tp_0', 'tp_10', 'tp_90', 'tp_50', 'tp_80', 'tp_max'], 'ORSIS3302DecoderForEnergy/hit')

            elif run<117 and cal==False:
                bkg_df = lh5.load_dfs(bkg_file_list, [f'{etype}', 'bl','bl_sig','A_10','AoE', 'ts_sec', 'dcr_raw', 'dcr_ftp', 'dcr_max', 'tp_0', 'tp_10', 'tp_90', 'tp_50', 'tp_80', 'tp_max'], 'ORSIS3302DecoderForEnergy/hit')
                alp_df = lh5.load_dfs(alp_file_list, [f'{etype}', 'bl','bl_sig','A_10','AoE', 'ts_sec', 'dcr_raw', 'dcr_ftp', 'dcr_max', 'tp_0', 'tp_10', 'tp_90', 'tp_50', 'tp_80', 'tp_max'], 'ORSIS3302DecoderForEnergy/hit')
            elif run>=117 and cal==False:
                bkg_df = lh5.load_dfs(bkg_file_list, [f'{etype}', 'bl','bl_sig', 'bl_slope', 'lf_max', 'A_10','AoE', 'dcr', 'tp_0', 'tp_10', 'tp_90', 'tp_50', 'tp_80', 'tp_max'], 'ORSIS3302DecoderForEnergy/hit')
                alp_df = lh5.load_dfs(alp_file_list, [f'{etype}', 'bl','bl_sig', 'bl_slope', 'lf_max', 'A_10','AoE', 'dcr', 'tp_0', 'tp_10', 'tp_90', 'tp_50', 'tp_80', 'tp_max'], 'ORSIS3302DecoderForEnergy/hit')

        elif hit==False:
            print('Using dsp files')
            bkg_file_list = bkg_lh5_dir + bkg_dg.fileDB['dsp_path'] + '/' + bkg_dg.fileDB['dsp_file']
            alp_file_list = alp_lh5_dir + alp_dg.fileDB['dsp_path'] + '/' + alp_dg.fileDB['dsp_file']
            if run<117 and cal==True:
                bkg_df = lh5.load_dfs(bkg_file_list, ['energy', 'trapEmax', 'trapEmax_cal', 'bl','bl_sig','A_10','AoE', 'ts_sec', 'dcr_raw', 'dcr_ftp', 'dcr_max', 'tp_0', 'tp_10', 'tp_90', 'tp_50', 'tp_80', 'tp_max'], 'ORSIS3302DecoderForEnergy/dsp')
                alp_df = lh5.load_dfs(alp_file_list, ['energy', 'trapEmax', 'trapEmax_cal', 'bl','bl_sig','A_10','AoE', 'ts_sec', 'dcr_raw', 'dcr_ftp', 'dcr_max', 'tp_0', 'tp_10', 'tp_90', 'tp_50', 'tp_80', 'tp_max'], 'ORSIS3302DecoderForEnergy/dsp')
            elif run>=117 and cal==True:
                bkg_df = lh5.load_dfs(bkg_file_list, ['energy', 'trapEmax', 'trapEftp', 'trapEmax_cal', 'trapEftp_cal', 'bl','bl_sig', 'bl_slope', 'lf_max', 'A_10','AoE', 'dcr', 'tp_0', 'tp_10', 'tp_90','tp_50', 'tp_80', 'tp_max'], 'ORSIS3302DecoderForEnergy/dsp')
                alp_df = lh5.load_dfs(alp_file_list, ['energy', 'trapEmax', 'trapEftp', 'trapEmax_cal', 'trapEftp_cal', 'bl','bl_sig', 'bl_slope', 'lf_max', 'A_10','AoE', 'dcr', 'tp_0', 'tp_10', 'tp_90','tp_50', 'tp_80', 'tp_max'], 'ORSIS3302DecoderForEnergy/dsp')

            elif run<117 and cal==False:
                bkg_df = lh5.load_dfs(bkg_file_list, [f'{etype}', 'bl','bl_sig','A_10','AoE', 'ts_sec', 'dcr_raw', 'dcr_ftp', 'dcr_max', 'tp_0', 'tp_10', 'tp_90', 'tp_50', 'tp_80', 'tp_max'], 'ORSIS3302DecoderForEnergy/dsp')
                alp_df = lh5.load_dfs(alp_file_list, [f'{etype}', 'bl','bl_sig','A_10','AoE', 'ts_sec', 'dcr_raw', 'dcr_ftp', 'dcr_max', 'tp_0', 'tp_10', 'tp_90', 'tp_50', 'tp_80', 'tp_max'], 'ORSIS3302DecoderForEnergy/dsp')
            elif run>=117 and cal==False:
                bkg_df = lh5.load_dfs(bkg_file_list, [f'{etype}', 'bl','bl_sig', 'bl_slope', 'lf_max', 'A_10','AoE', 'dcr', 'tp_0', 'tp_10', 'tp_90', 'tp_50', 'tp_80', 'tp_max'], 'ORSIS3302DecoderForEnergy/dsp')
                alp_df = lh5.load_dfs(alp_file_list, [f'{etype}', 'bl','bl_sig', 'bl_slope', 'lf_max', 'A_10','AoE', 'dcr', 'tp_0', 'tp_10', 'tp_90', 'tp_50', 'tp_80', 'tp_max'], 'ORSIS3302DecoderForEnergy/dsp')

        else:
            print('dont know what to do here! need to specify if working with calibrated/uncalibrated data, or dsp/hit files')



        # use baseline cut
        if run <79:
            bl_cut_lo, bl_cut_hi = 9150,9320
        if run>79 and run <117:
            bl_cut_lo, bl_cut_hi = 8500, 10000
        if run>=117:
            bl_cut_lo, bl_cut_hi = 9700, 9760

        bkg_df_cut = bkg_df.query(f'bl > {bl_cut_lo} and bl < {bl_cut_hi}').copy()
        alp_df_cut = alp_df.query(f'bl > {bl_cut_lo} and bl < {bl_cut_hi}').copy()

        #creat new DCR
        if run <= 86:
            const = 0.0002
            bkg_df_cut['dcr_corr'] = bkg_df_cut['dcr'] + const*bkg_df_cut['trapEftp']
            alp_df_cut['dcr_corr'] = alp_df_cut['dcr'] + const*alp_df_cut['trapEftp']

        if run>86 and run <117:
            const = -0.0225
            bkg_df_cut['dcr_corr'] = bkg_df_cut['dcr_raw'] + const*bkg_df_cut['trapEmax']
            alp_df_cut['dcr_corr'] = alp_df_cut['dcr_raw'] + const*alp_df_cut['trapEmax']

        if run>=117:
            const = -0.0003
            const2 = -0.0000003
            bkg_df_cut['dcr_corr'] = bkg_df_cut['dcr'] + const*(bkg_df_cut['trapEftp']) + const2*(bkg_df_cut['trapEftp'])**2
            alp_df_cut['dcr_corr'] = alp_df_cut['dcr'] + const*(alp_df_cut['trapEftp']) + const2*(alp_df_cut['trapEftp'])**2
            # if cal==True:
            #     #creat new DCR
            #     const = -0.0015
            #     const2 = -0.0000015
            #     df_cut['dcr_linoff'] = df_cut['dcr'] + const*(df_cut['trapEftp_cal']) + const2*(df_cut['trapEftp_cal'])**2



        #create 0-50
        bkg_df_cut['tp0_50'] = bkg_df_cut['tp_50']- bkg_df_cut['tp_0']
        alp_df_cut['tp0_50'] = alp_df_cut['tp_50']- alp_df_cut['tp_0']

        # create cut for alphas
        # alpha_cut = 'dcr_linoff > 25 and dcr_linoff < 200 and tp0_50 > 100 and tp0_50 < 400 and trapEmax_cal < 6000'
        # new_dcr_cut = df_cut.query(alpha_cut).copy()
        # len(new_dcr_cut)

        #-------------------------------------
        # Plots before alpha cuts
        #--------------------

        # select energy type and energy range
        if cal==False:
            elo, ehi, epb = 0, 10000, 10 #entire enerty range trapEftp
            e_unit = ' (uncal)'
        elif cal==True:
            elo, ehi, epb = 0., 6000., 5.
            etype=etype_cal
            e_unit = ' (keV)'

        # Make background-subtracted (calibrated) energy spectrum_________
        # Background run spectrum
        nbx = int((ehi-elo)/epb)
        
        bkg_ene_hist, bins = np.histogram(bkg_df_cut[etype], bins=nbx, range=([elo, ehi]))
        bkg_ene_hist_norm = np.divide(bkg_ene_hist, (bkg_rt_min))

        # Alpha run spectrum
        alp_ene_hist, bins = np.histogram(alp_df_cut[etype], bins=nbx, range=([elo, ehi]))
        alp_ene_hist_norm = np.divide(alp_ene_hist, (alp_rt_min))

        # Background subtracted spectrum_________
        energy_bgSub_hist = alp_ene_hist_norm - bkg_ene_hist_norm


        fig, ax = plt.subplots()
        fig.suptitle(f'Background-Subtracted Energy', horizontalalignment='center', fontsize=16)

        plt.semilogy(bins[1:], energy_bgSub_hist, ds='steps', c='b', lw=1) #, label=f'{etype}'

        ax.set_xlabel(f'{etype+e_unit}', fontsize=16)
        ax.set_ylabel(f'counts/min/{str(epb)}{e_unit}', fontsize=16)
        plt.ylim(0.001, 20)
        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)

        ax.text(0.95, 0.83, f'r = {radius} mm \ntheta = {angle_det} deg', verticalalignment='bottom',
                    horizontalalignment='right', transform=ax.transAxes, color='green', fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

        # plt.legend()
        plt.title(f'\n{alp_runtype} run {run}, {alp_rt_min:.2f} mins \nbkg run {bkg}, {bkg_rt_min:.2f} mins', fontsize=12)
        plt.tight_layout()
        # plt.savefig(f'./plots/normScan/cal_normScan/{runtype}_energy_run{run}.png', dpi=200)
        if alp_runtype=='alp':
            plt.savefig(f'./plots/angleScan/bgSub_energy_{radius}mm_{angle_det}deg_{alp_runtype}_run{run}.png', dpi=200)
        elif alp_runtype=='bkg':
            plt.savefig(f'./plots/angleScan/bgSub_energy_{alp_runtype}_run{run}.png', dpi=200)
        plt.clf()
        plt.close()


        # AoE vs E---------------------------------
        # normalized by runtime
        fig, ax = plt.subplots()
        alo, ahi, apb = 0.0, 0.09, 0.0001
        if run>=60:
            alo, ahi, apb = 0.005, 0.0905, 0.0001
        if run>=117:
            alo, ahi, apb = 0.0, 0.15, 0.00015

        nbx = int((ehi-elo)/epb)
        nby = int((ahi-alo)/apb)

        fig.suptitle(f'Background-Subtracted A/E vs Energy', horizontalalignment='center', fontsize=16)

        bkg_aoe_hist, xedges, yedges = np.histogram2d(bkg_df_cut[etype], bkg_df_cut['AoE'], bins=[nbx, nby], range=([elo, ehi], [alo, ahi]))
        alp_aoe_hist, xedges, yedges = np.histogram2d(alp_df_cut[etype], alp_df_cut['AoE'], bins=[nbx, nby], range=([elo, ehi], [alo, ahi]))
        X, Y = np.mgrid[elo:ehi:nbx*1j, alo:ahi:nby*1j]

        bkg_aoe_hist_norm = np.divide(bkg_aoe_hist, (bkg_rt_min))
        alp_aoe_hist_norm = np.divide(alp_aoe_hist, (alp_rt_min))

        AoE_bgSub_hist = alp_aoe_hist_norm - bkg_aoe_hist_norm

        pcm = plt.pcolormesh(X, Y, AoE_bgSub_hist, norm=LogNorm(0.002, 0.2))

        cb = plt.colorbar()
        cb.set_label("counts/min", ha = 'right', va='center', rotation=270, fontsize=14)
        cb.ax.tick_params(labelsize=12)
        ax.set_xlabel(f'{etype+e_unit} \n({str(epb)}{e_unit} bins)', fontsize=16)
        ax.set_ylabel('A/E (arb)', fontsize=16)
        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)


        ax.text(0.95, 0.80, f'r = {radius} mm \ntheta = {angle_det} deg', verticalalignment='bottom',
                    horizontalalignment='right', transform=ax.transAxes, color='green', fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

        # plt.legend()
        plt.title(f'\n{alp_runtype} run {run}, {alp_rt_min:.2f} mins \nbkg run {bkg}, {bkg_rt_min:.2f} mins', fontsize=12)
        plt.tight_layout()
        # plt.savefig(f'./plots/normScan/cal_normScan/{runtype}_AoE_run{run}.png', dpi=200)
        if alp_runtype=='alp':
            plt.savefig(f'./plots/angleScan/bgSub_AoE_{radius}mm_{angle_det}deg_{alp_runtype}_run{run}.png', dpi=200)
        elif alp_runtype=='bkg':
            plt.savefig(f'./plots/angleScan/bgSub_AoE_{alp_runtype}_run{run}.png', dpi=200)
        # plt.show()

        plt.clf()
        plt.close()

        # DCR vs E___________

        fig, ax = plt.subplots()

        if run>=60 and run<117:
            dlo, dhi, dpb = -100, 300, 0.6
        elif run>=117:
            dlo, dhi, dpb = -20., 40, 0.1

        nbx = int((ehi-elo)/epb)
        nby = int((dhi-dlo)/dpb)

        fig.suptitle(f'DCR vs Energy', horizontalalignment='center', fontsize=16)

        bkg_dcr_hist, xedges, yedges = np.histogram2d(bkg_df_cut[etype], bkg_df_cut['dcr_corr'], bins=[nbx, nby], range=([elo, ehi], [dlo, dhi]))
        alp_dcr_hist, xedges, yedges = np.histogram2d(alp_df_cut[etype], alp_df_cut['dcr_corr'], bins=[nbx, nby], range=([elo, ehi], [dlo, dhi]))
        X, Y = np.mgrid[elo:ehi:nbx*1j, dlo:dhi:nby*1j]

        bkg_dcr_hist_norm = np.divide(bkg_dcr_hist, (bkg_rt_min))
        alp_dcr_hist_norm = np.divide(alp_dcr_hist, (alp_rt_min))

        dcr_bgSub_hist = alp_dcr_hist_norm - bkg_dcr_hist_norm

        pcm = plt.pcolormesh(X, Y, dcr_bgSub_hist, norm=LogNorm(0.002, 0.2))

        cb = plt.colorbar()
        cb.set_label("counts/min", ha = 'right', va='center', rotation=270, fontsize=14)
        cb.ax.tick_params(labelsize=12)
        ax.set_xlabel(f'Energy (keV) \n({str(epb)}{e_unit} bins)', fontsize=16)
        ax.set_ylabel('DCR (arb)', fontsize=16)
        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)

        # plt.legend()

        ax.text(0.95, 0.80, f'r = {radius} mm \ntheta = {angle_det} deg', verticalalignment='bottom',
                    horizontalalignment='right', transform=ax.transAxes, color='green', fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

        plt.title(f'\n{alp_runtype} run {run}, {alp_rt_min:.2f} mins \nbkg run {bkg}, {bkg_rt_min:.2f} mins', fontsize=12)
        plt.tight_layout()
        # plt.savefig(f'./plots/normScan/cal_normScan/{runtype}_dcr_run{run}.png', dpi=200)
        if alp_runtype=='alp':
            plt.savefig(f'./plots/angleScan/bgSub_DCR_{radius}mm_{angle_det}deg_{alp_runtype}_run{run}.png', dpi=200)
        elif alp_runtype=='bkg':
            plt.savefig(f'./plots/angleScan/bgSub_DCR_{alp_runtype}_run{run}.png', dpi=200)
        # plt.show()
        plt.clf()
        plt.close()

        # DCR vs A/E___________

        fig, ax = plt.subplots()
        nbx = int((ahi-alo)/apb)
        nby = int((dhi-dlo)/dpb)

        fig.suptitle(f'Background-Subtracted A/E vs DCR', horizontalalignment='center', fontsize=16)

        bkg_aoeVdcr_hist, xedges, yedges = np.histogram2d(bkg_df_cut['AoE'], bkg_df_cut['dcr_corr'], bins=[nbx, nby], range=([alo, ahi], [dlo, dhi]))
        alp_aoeVdcr_hist, xedges, yedges = np.histogram2d(alp_df_cut['AoE'], alp_df_cut['dcr_corr'], bins=[nbx, nby], range=([alo, ahi], [dlo, dhi]))
        X, Y = np.mgrid[alo:ahi:nbx*1j, dlo:dhi:nby*1j]

        bkg_aoeVdcr_hist_norm = np.divide(bkg_aoeVdcr_hist, (bkg_rt_min))
        alp_aoeVdcr_hist_norm = np.divide(alp_aoeVdcr_hist, (alp_rt_min))

        aoeVdcr_bgSub_hist = alp_aoeVdcr_hist_norm - bkg_aoeVdcr_hist_norm

        pcm = plt.pcolormesh(X, Y, aoeVdcr_bgSub_hist, norm=LogNorm(0.002, 0.2))

        cb = plt.colorbar()
        cb.set_label("counts/min", ha = 'right', va='center', rotation=270, fontsize=14)
        cb.ax.tick_params(labelsize=12)
        ax.set_xlabel('A/E (arb)', fontsize=16)
        ax.set_ylabel('DCR (arb)', fontsize=16)
        plt.setp(ax.get_xticklabels(), fontsize=12)
        plt.setp(ax.get_yticklabels(), fontsize=14)

        # plt.legend()
        ax.text(0.95, 0.80, f'r = {radius} mm \ntheta = {angle_det} deg', verticalalignment='bottom',
                    horizontalalignment='right', transform=ax.transAxes, color='green', fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

        plt.title(f'\n{alp_runtype} run {run}, {alp_rt_min:.2f} mins \nbkg run {bkg}, {bkg_rt_min:.2f} mins', fontsize=12)
        plt.tight_layout()
        # plt.savefig(f'./plots/normScan/cal_normScan/{runtype}_AoE_vs_dcr_run{run}.png', dpi=200)
        if alp_runtype=='alp':
            plt.savefig(f'./plots/angleScan/bgSub_aoeVdcr_{radius}mm_{angle_det}deg_{alp_runtype}_run{run}.png', dpi=200)
        elif alp_runtype=='bkg':
            plt.savefig(f'./plots/angleScan/bgSub_aoeVdcr_{alp_runtype}_run{run}.png', dpi=200)
        # plt.show()
        plt.clf()
        plt.close()

        # DCR vs tp_50___________

        fig, ax = plt.subplots()
        fig.suptitle(f'Background-Subtracted DCR vs 50% rise time', horizontalalignment='center', fontsize=16)

        tlo, thi, tpb = 0, 400, 10

        nbx = int((dhi-dlo)/dpb)
        nby = int((thi-tlo)/tpb)

        bkg_DCRvTp050_hist, xedges, yedges = np.histogram2d(bkg_df_cut['dcr_corr'], bkg_df_cut['tp0_50'], bins=[nbx, nby], range=([dlo, dhi], [tlo, thi]))
        alp_DCRvTp050_hist, xedges, yedges = np.histogram2d(alp_df_cut['dcr_corr'], alp_df_cut['tp0_50'], bins=[nbx, nby], range=([dlo, dhi], [tlo, thi]))
        X, Y = np.mgrid[dlo:dhi:nbx*1j, tlo:thi:nby*1j]

        bkg_DCRvTp050_hist_norm = np.divide(bkg_DCRvTp050_hist, (bkg_rt_min))
        alp_DCRvTp050_hist_norm = np.divide(alp_DCRvTp050_hist, (alp_rt_min))

        DCRvTp050_bgSub_hist = alp_DCRvTp050_hist_norm - bkg_DCRvTp050_hist_norm

        pcm = plt.pcolormesh(X, Y, DCRvTp050_bgSub_hist, norm=LogNorm(0.002, 0.2))

        cb = plt.colorbar()
        cb.set_label("counts/min", ha = 'right', va='center', rotation=270, fontsize=14)
        cb.ax.tick_params(labelsize=12)
        ax.set_xlabel('DCR (arb)', fontsize=16)
        ax.set_ylabel('tp 0-50 (ns)', fontsize=16)
        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)

        # plt.legend()
        ax.text(0.95, 0.80, f'r = {radius} mm \ntheta = {angle_det} deg', verticalalignment='bottom',
                    horizontalalignment='right', transform=ax.transAxes, color='green', fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.95, 'pad': 10})

        plt.title(f'\n{alp_runtype} run {run}, {alp_rt_min:.2f} mins \nbkg run {bkg}, {bkg_rt_min:.2f} mins', fontsize=12)
        plt.tight_layout()
        # plt.savefig(f'./plots/normScan/cal_normScan/{runtype}_dcr_vs_tp0_50_run{run}.png', dpi=200)
        if alp_runtype=='alp':
            plt.savefig(f'./plots/angleScan/bgSub_DCRvTp050_{radius}mm_{angle_det}deg_{alp_runtype}_run{run}.png', dpi=200)
        elif alp_runtype=='bkg':
            plt.savefig(f'./plots/angleScan/bgSub_DCRvTp050_{alp_runtype}_run{run}.png', dpi=200)
        # plt.show()
        plt.clf()
        plt.close()

def normalized_dcr_AvE(runs, user=False, hit=True, cal=True, etype='trapEmax', cut=True, campaign=''):

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
            #radius = int(radius)
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
            if run<79 and cal==True:
                df = lh5.load_dfs(file_list, ['energy', 'trapEmax', 'trapEftp', 'trapEftp_cal', 'bl','bl_sig', 'bl_slope', 'lf_max', 'A_10','AoE', 'dcr', 'tp_0', 'tp_10', 'tp_90', 'tp_50', 'tp_80', 'tp_max'], 'ORSIS3302DecoderForEnergy/hit')
            elif run>79 and run <117 and cal==True:
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



        # use baseline cut
        if run <79:
            bl_cut_lo, bl_cut_hi = 9150,9320
        if run>79 and run <117:
            bl_cut_lo, bl_cut_hi = 8500, 10000
        if run>=117:
            bl_cut_lo, bl_cut_hi = 9700, 9760

        df_cut = df.query(f'bl > {bl_cut_lo} and bl < {bl_cut_hi}').copy()

        #creat new DCR
        if run>35 and run<=56:
            const = const = 0.0011
            df_cut['dcr_linoff'] = df_cut['dcr'] + const*df_cut['trapEftp']
        if run>56 and run< 79:
            const = 0.0002
            df_cut['dcr_linoff'] = df_cut['dcr'] + const*df_cut['trapEftp']
        if run>79 and run <= 86:
            const = 0.0555
            df_cut['dcr_linoff'] = df_cut['dcr_raw'] + const*df_cut['trapEmax']

        if run>86 and run <117:
            const = -0.0225
            df_cut['dcr_linoff'] = df_cut['dcr_raw'] + const*df_cut['trapEmax']

        if run>=117:
            const = -0.0003
            const2 = -0.0000003
            df_cut['dcr_linoff'] = df_cut['dcr'] + const*(df_cut['trapEftp']) + const2*(df_cut['trapEftp'])**2
            # if cal==True:
            #     #creat new DCR
            #     const = -0.0015
            #     const2 = -0.0000015
            #     df_cut['dcr_linoff'] = df_cut['dcr'] + const*(df_cut['trapEftp_cal']) + const2*(df_cut['trapEftp_cal'])**2



        #create 0-50
        df_cut['tp0_50'] = df_cut['tp_50']- df_cut['tp_0']

        # create cut for alphas
        # alpha_cut = 'dcr_linoff > 25 and dcr_linoff < 200 and tp0_50 > 100 and tp0_50 < 400 and trapEmax_cal < 6000'
        # new_dcr_cut = df_cut.query(alpha_cut).copy()
        # len(new_dcr_cut)

        #-------------------------------------
        # Plots before alpha cuts
        #--------------------

        # select energy type and energy range
        if cal==False:
            elo, ehi, epb = 0, 10000, 10 #entire enerty range trapEftp
            e_unit = ' (uncal)'
        elif cal==True:
            elo, ehi, epb = 0, 6000, 2
            etype=etype_cal
            e_unit = ' (keV)'

        # Make (calibrated) energy spectrum_________

        fig, ax = plt.subplots()
        fig.suptitle(f'Energy', horizontalalignment='center', fontsize=16)

        nbx = int((ehi-elo)/epb)

        energy_hist, bins = np.histogram(df_cut[etype], bins=nbx,
                range=[elo, ehi])
        energy_rt = np.divide(energy_hist, rt_min * 60)

        plt.semilogy(bins[1:], energy_rt, ds='steps', c='b', lw=1) #, label=f'{etype}'

        ax.set_xlabel(f'{etype+e_unit}', fontsize=16)
        ax.set_ylabel('counts/sec', fontsize=16)
        plt.ylim(0.0001,2)
        plt.xlim(10., ehi)
        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)

        ax.text(0.95, 0.83, f'r = {radius} mm \ntheta = {angle_det} deg', verticalalignment='bottom',
                    horizontalalignment='right', transform=ax.transAxes, color='green', fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

        # plt.legend()
        plt.title(f'\n{runtype} run {run}, {rt_min:.2f} mins', fontsize=12)
        plt.tight_layout()
        # plt.savefig(f'./plots/normScan/cal_normScan/{runtype}_energy_run{run}.png', dpi=200)
        if runtype=='alp':
            plt.savefig(f'./plots/{campaign}normalized_{runtype}_energy_{radius}mm_{angle_det}deg_run{run}.png', dpi=200)
        elif runtype=='bkg':
            plt.savefig(f'./plots/{campaign}normalized_{runtype}_energy_run{run}.png', dpi=200)
        
        #now zoom into 60 keV
        plt.xlim(40, 80)
        plt.ylim(0.1, 1)
        
        if runtype=='alp':
            plt.savefig(f'./plots/{campaign}normalized_{runtype}_energy_60keV_{radius}mm_{angle_det}deg_run{run}.png', dpi=200)
        elif runtype=='bkg':
            plt.savefig(f'./plots/{campaign}normalized_{runtype}_energy_60keV_run{run}.png', dpi=200)
        plt.clf()
        plt.close()


        # AoE vs E---------------------------------
        # normalized by runtime
        fig, ax = plt.subplots()
        alo, ahi, apb = 0.0, 0.09, 0.0001
        if run>=36:
            alo, ahi, apb = 0.005, 0.075, 0.0001
        if run>=117:
            alo, ahi, apb = 0.0, 0.15, 0.00015

        nbx = int((ehi-elo)/epb)
        nby = int((ahi-alo)/apb)

        fig.suptitle(f'A/E vs Energy', horizontalalignment='center', fontsize=16)

        aoe_hist, xedges, yedges = np.histogram2d(df_cut[etype], df_cut['AoE'], bins=[nbx, nby], range=([elo, ehi], [alo, ahi]))
        X, Y = np.mgrid[elo:ehi:nbx*1j, alo:ahi:nby*1j]

        aoe_hist_norm = np.divide(aoe_hist, (rt_min))

        pcm = plt.pcolormesh(X, Y, aoe_hist_norm, norm=LogNorm(0.002, 0.2))

        cb = plt.colorbar()
        cb.set_label("counts/min", ha = 'right', va='center', rotation=270, fontsize=14)
        cb.ax.tick_params(labelsize=12)
        ax.set_xlabel(f'{etype+e_unit}', fontsize=16)
        ax.set_ylabel('A/E (arb)', fontsize=16)
        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)


        ax.text(0.95, 0.83, f'r = {radius} mm \ntheta = {angle_det} deg', verticalalignment='bottom',
                    horizontalalignment='right', transform=ax.transAxes, color='green', fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

        # plt.legend()
        plt.title(f'\n{runtype} run {run}, {rt_min:.2f} mins', fontsize=12)
        plt.tight_layout()
        # plt.savefig(f'./plots/normScan/cal_normScan/{runtype}_AoE_run{run}.png', dpi=200)
        if runtype=='alp':
            plt.savefig(f'./plots/{campaign}normalized_{runtype}_AoE_{radius}mm_{angle_det}deg_run{run}.png', dpi=200)
        elif runtype=='bkg':
            plt.savefig(f'./plots/{campaign}normalized_{runtype}_normalized_AoE_run{run}.png', dpi=200)
        # plt.show()

        plt.clf()
        plt.close()

        # DCR vs E___________

        fig, ax = plt.subplots()

        if run>=36 and run<117:
            dlo, dhi, dpb = -20., 60, 0.1
        elif run>=117:
            dlo, dhi, dpb = -20., 40, 0.1

        nbx = int((ehi-elo)/epb)
        nby = int((dhi-dlo)/dpb)

        fig.suptitle(f'DCR vs Energy', horizontalalignment='center', fontsize=16)

        dcr_hist, xedges, yedges = np.histogram2d(df_cut[etype], df_cut['dcr_linoff'], bins=[nbx, nby], range=([elo, ehi], [dlo, dhi]))
        X, Y = np.mgrid[elo:ehi:nbx*1j, dlo:dhi:nby*1j]

        dcr_hist_norm = np.divide(dcr_hist, (rt_min))

        pcm = plt.pcolormesh(X, Y, dcr_hist_norm, norm=LogNorm(0.002, 0.2))

        cb = plt.colorbar()
        cb.set_label("counts/min", ha = 'right', va='center', rotation=270, fontsize=14)
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
        # plt.savefig(f'./plots/normScan/cal_normScan/{runtype}_dcr_run{run}.png', dpi=200)
        if runtype=='alp':
            plt.savefig(f'./plots/{campaign}normalized_{runtype}_DCR_{radius}mm_{angle_det}deg_run{run}.png', dpi=200)
        elif runtype=='bkg':
            plt.savefig(f'./plots/{campaign}normalized_{runtype}_DCR_run{run}.png', dpi=200)
        # plt.show()
        plt.clf()
        plt.close()

        # DCR vs A/E___________

        fig, ax = plt.subplots()
        nbx = int((ahi-alo)/apb)
        nby = int((dhi-dlo)/dpb)

        fig.suptitle(f'A/E vs DCR', horizontalalignment='center', fontsize=16)

        aoeVdcr_hist, xedges, yedges = np.histogram2d(df_cut['AoE'], df_cut['dcr_linoff'], bins=[nbx, nby], range=([alo, ahi], [dlo, dhi]))
        X, Y = np.mgrid[alo:ahi:nbx*1j, dlo:dhi:nby*1j]

        aoeVdcr_hist_norm = np.divide(aoeVdcr_hist, (rt_min))

        pcm = plt.pcolormesh(X, Y, aoeVdcr_hist_norm, norm=LogNorm(0.002, 0.2))

        cb = plt.colorbar()
        cb.set_label("counts/min", ha = 'right', va='center', rotation=270, fontsize=14)
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
        # plt.savefig(f'./plots/normScan/cal_normScan/{runtype}_AoE_vs_dcr_run{run}.png', dpi=200)
        if runtype=='alp':
            plt.savefig(f'./plots/{campaign}normalized_{runtype}_AoEvDCR_{radius}mm_{angle_det}deg_run{run}.png', dpi=200)
        elif runtype=='bkg':
            plt.savefig(f'./plots/{campaign}normalized_{runtype}_AoEvDCR_run{run}.png', dpi=200)
        # plt.show()
        plt.clf()
        plt.close()

        # DCR vs tp_50___________

        fig, ax = plt.subplots()
        fig.suptitle(f'DCR vs 50% rise time', horizontalalignment='center', fontsize=16)

        tlo, thi, tpb = 0, 400, 10

        nbx = int((dhi-dlo)/dpb)
        nby = int((thi-tlo)/tpb)

        DCRvTp050_hist, xedges, yedges = np.histogram2d(df_cut['dcr_linoff'], df_cut['tp0_50'], bins=[nbx, nby], range=([dlo, dhi], [tlo, thi]))
        X, Y = np.mgrid[dlo:dhi:nbx*1j, tlo:thi:nby*1j]

        DCRvTp050_hist_norm = np.divide(DCRvTp050_hist, (rt_min))

        pcm = plt.pcolormesh(X, Y, DCRvTp050_hist_norm, norm=LogNorm(0.002, 0.2))

        cb = plt.colorbar()
        cb.set_label("counts/min", ha = 'right', va='center', rotation=270, fontsize=14)
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
        # plt.savefig(f'./plots/normScan/cal_normScan/{runtype}_dcr_vs_tp0_50_run{run}.png', dpi=200)
        if runtype=='alp':
            plt.savefig(f'./plots/{campaign}normalized_{runtype}_DCRvTp050_{radius}mm_{angle_det}deg_run{run}.png', dpi=200)
        elif runtype=='bkg':
            plt.savefig(f'./plots/{campaign}normalized_{runtype}_DCRvTp050_run{run}.png', dpi=200)
        # plt.show()
        plt.clf()
        plt.close()


def dcr_AvE(runs, user=False, hit=True, cal=True, etype='trapEmax', cut=True):

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
            if run<=117 and cal==True:
                df = lh5.load_dfs(file_list, ['energy', 'trapEmax', 'trapEmax_cal', 'bl','bl_sig','A_10','AoE', 'ts_sec', 'dcr_raw', 'dcr_ftp', 'dcr_max', 'tp_0','tp_10', 'tp_90', 'tp_50', 'tp_80', 'tp_max'], 'ORSIS3302DecoderForEnergy/hit')
            elif run>117 and cal==True:
                df = lh5.load_dfs(file_list, ['energy', 'trapEmax', 'trapEftp', 'trapEmax_cal', 'trapEftp_cal', 'bl','bl_sig', 'bl_slope', 'lf_max', 'A_10','AoE', 'dcr', 'tp_0', 'tp_10', 'tp_90', 'tp_50', 'tp_80', 'tp_max'], 'ORSIS3302DecoderForEnergy/hit')

            elif run<=117 and cal==False:
                df = lh5.load_dfs(file_list, [f'{etype}', 'bl','bl_sig','A_10','AoE', 'ts_sec', 'dcr_raw', 'dcr_ftp', 'dcr_max', 'tp_0', 'tp_10', 'tp_90', 'tp_50', 'tp_80', 'tp_max'], 'ORSIS3302DecoderForEnergy/hit')
            elif run>117 and cal==False:
                df = lh5.load_dfs(file_list, [f'{etype}', 'bl','bl_sig', 'bl_slope', 'lf_max', 'A_10','AoE', 'dcr', 'tp_0', 'tp_10', 'tp_90', 'tp_50', 'tp_80', 'tp_max'], 'ORSIS3302DecoderForEnergy/hit')

        elif hit==False:
            print('Using dsp files')
            file_list = lh5_dir + dg.fileDB['dsp_path'] + '/' + dg.fileDB['dsp_file']
            if run<=117 and cal==True:
                df = lh5.load_dfs(file_list, [f'{etype}', f'{etype_cal}', 'bl','bl_sig','A_10','AoE', 'ts_sec', 'dcr_raw', 'dcr_ftp', 'dcr_max', 'tp_0', 'tp_10', 'tp_90', 'tp_50', 'tp_80', 'tp_max'], 'ORSIS3302DecoderForEnergy/dsp')
            elif run>117 and cal==True:
                df = lh5.load_dfs(file_list, [f'{etype}', f'{etype_cal}', 'bl','bl_sig', 'bl_slope', 'lf_max', 'A_10','AoE', 'dcr', 'tp_0', 'tp_10', 'tp_90','tp_50', 'tp_80', 'tp_max'], 'ORSIS3302DecoderForEnergy/dsp')

            elif run<=117 and cal==False:
                df = lh5.load_dfs(file_list, [f'{etype}', 'bl','bl_sig','A_10','AoE', 'ts_sec', 'dcr_raw', 'dcr_ftp', 'dcr_max', 'tp_0', 'tp_10', 'tp_90', 'tp_50', 'tp_80', 'tp_max'], 'ORSIS3302DecoderForEnergy/dsp')
            elif run>117 and cal==False:
                df = lh5.load_dfs(file_list, [f'{etype}', 'bl','bl_sig', 'bl_slope', 'lf_max', 'A_10','AoE', 'dcr', 'tp_0', 'tp_10', 'tp_90', 'tp_50', 'tp_80', 'tp_max'], 'ORSIS3302DecoderForEnergy/dsp')

        else:
            print('dont know what to do here! need to specify if working with calibrated/uncalibrated data, or dsp/hit files')



        # use baseline cut
        if run <=117:
            bl_cut_lo, bl_cut_hi = 8500, 10000
        if run>117:
            bl_cut_lo, bl_cut_hi = 9700, 9760

        df_cut = df.query(f'bl > {bl_cut_lo} and bl < {bl_cut_hi}').copy()

        #creat new DCR
        if run <= 86:
            const = 0.0555
            df_cut['dcr_linoff'] = df_cut['dcr_raw'] + const*df_cut['trapEmax']

        if run>86 and run <=117:
            const = -0.0225
            df_cut['dcr_linoff'] = df_cut['dcr_raw'] + const*df_cut['trapEmax']

        if run>117:
            const = -0.0003
            const2 = -0.0000003
            df_cut['dcr_linoff'] = df_cut['dcr'] + const*(df_cut['trapEftp']) + const2*(df_cut['trapEftp'])**2
            # if cal==True:
            #     #creat new DCR
            #     const = -0.0015
            #     const2 = -0.0000015
            #     df_cut['dcr_linoff'] = df_cut['dcr'] + const*(df_cut['trapEftp_cal']) + const2*(df_cut['trapEftp_cal'])**2



        #create 0-50
        df_cut['tp0_50'] = df_cut['tp_50']- df_cut['tp_0']

        # create cut for alphas
        # alpha_cut = 'dcr_linoff > 25 and dcr_linoff < 200 and tp0_50 > 100 and tp0_50 < 400 and trapEmax_cal < 6000'
        # new_dcr_cut = df_cut.query(alpha_cut).copy()
        # len(new_dcr_cut)

        #-------------------------------------
        # Plots before alpha cuts
        #--------------------

        # select energy type and energy range
        if cal==False:
            elo, ehi, epb = 0, 10000, 10 #entire enerty range trapEftp
            e_unit = ' (uncal)'
        elif cal==True:
            elo, ehi, epb = 0, 6000, 10
            etype=etype_cal
            e_unit = ' (keV)'

        # Make (calibrated) energy spectrum_________

        fig, ax = plt.subplots()
        fig.suptitle(f'Energy', horizontalalignment='center', fontsize=16)

        nbx = int((ehi-elo)/epb)

        energy_hist, bins = np.histogram(df_cut[etype], bins=nbx,
                range=[elo, ehi])
        energy_rt = np.divide(energy_hist, rt_min * 60)

        plt.semilogy(bins[1:], energy_rt, ds='steps', c='b', lw=1) #, label=f'{etype}'

        ax.set_xlabel(f'{etype+e_unit}', fontsize=16)
        ax.set_ylabel('counts/sec', fontsize=16)
        plt.ylim(0.0001,5)
        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)

        ax.text(0.95, 0.83, f'r = {radius} mm \ntheta = {angle_det} deg', verticalalignment='bottom',
                    horizontalalignment='right', transform=ax.transAxes, color='green', fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

        # plt.legend()
        plt.title(f'\n{runtype} run {run}, {rt_min:.2f} mins', fontsize=12)
        plt.tight_layout()
        # plt.savefig(f'./plots/normScan/cal_normScan/{runtype}_energy_run{run}.png', dpi=200)
        if runtype=='alp':
            plt.savefig(f'./plots/angleScan/{runtype}_energy_{radius}mm_{angle_det}deg_run{run}.png', dpi=200)
        elif runtype=='bkg':
            plt.savefig(f'./plots/angleScan/{runtype}_energy_run{run}.png', dpi=200)
        plt.clf()
        plt.close()


        # AoE vs E---------------------------------
        fig, ax = plt.subplots()
        alo, ahi, apb = 0.0, 0.09, 0.0001
        if run>=60:
            alo, ahi, apb = 0.005, 0.0905, 0.0001
        if run>117:
            alo, ahi, apb = 0.0, 0.125, 0.001

        nbx = int((ehi-elo)/epb)
        nby = int((ahi-alo)/apb)

        fig.suptitle(f'A/E vs Energy', horizontalalignment='center', fontsize=16)

        h = plt.hist2d(df_cut[etype], df_cut['AoE'], bins=[nbx,nby],
                    range=[[elo, ehi], [alo, ahi]], cmap='viridis', norm=LogNorm())

        cb = plt.colorbar()
        cb.set_label("counts", ha = 'right', va='center', rotation=270, fontsize=14)
        cb.ax.tick_params(labelsize=12)
        ax.set_xlabel(f'{etype+e_unit}', fontsize=16)
        ax.set_ylabel('A/E (arb)', fontsize=16)
        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)


        ax.text(0.95, 0.83, f'r = {radius} mm \ntheta = {angle_det} deg', verticalalignment='bottom',
                    horizontalalignment='right', transform=ax.transAxes, color='green', fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

        # plt.legend()
        plt.title(f'\n{runtype} run {run}, {rt_min:.2f} mins', fontsize=12)
        plt.tight_layout()
        # plt.savefig(f'./plots/normScan/cal_normScan/{runtype}_AoE_run{run}.png', dpi=200)
        if runtype=='alp':
            plt.savefig(f'./plots/angleScan/{runtype}_AoE_{radius}mm_{angle_det}deg_run{run}.png', dpi=200)
        elif runtype=='bkg':
            plt.savefig(f'./plots/angleScan/{runtype}_AoE_run{run}.png', dpi=200)
        # plt.show()

        plt.clf()
        plt.close()

        # DCR vs E___________

        fig, ax = plt.subplots()

        if run>=60 and run<117:
            dlo, dhi, dpb = -100, 300, 0.6
        elif run>117:
            dlo, dhi, dpb = -20., 60, 0.1

        nbx = int((ehi-elo)/epb)
        nby = int((dhi-dlo)/dpb)

        fig.suptitle(f'DCR vs Energy', horizontalalignment='center', fontsize=16)

        h = plt.hist2d(df_cut[etype], df_cut['dcr_linoff'], bins=[nbx,nby],
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
        # plt.savefig(f'./plots/normScan/cal_normScan/{runtype}_dcr_run{run}.png', dpi=200)
        if runtype=='alp':
            plt.savefig(f'./plots/angleScan/{runtype}_DCR_{radius}mm_{angle_det}deg_run{run}.png', dpi=200)
        elif runtype=='bkg':
            plt.savefig(f'./plots/angleScan/{runtype}_DCR_run{run}.png', dpi=200)
        # plt.show()
        plt.clf()
        plt.close()

        # DCR vs A/E___________

        fig, ax = plt.subplots()
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
        # plt.savefig(f'./plots/normScan/cal_normScan/{runtype}_AoE_vs_dcr_run{run}.png', dpi=200)
        if runtype=='alp':
            plt.savefig(f'./plots/angleScan/{runtype}_AoEvDCR_{radius}mm_{angle_det}deg_run{run}.png', dpi=200)
        elif runtype=='bkg':
            plt.savefig(f'./plots/angleScan/{runtype}_AoEvDCR_run{run}.png', dpi=200)
        # plt.show()
        plt.clf()
        plt.close()

        # DCR vs tp_50___________

        fig, ax = plt.subplots()
        fig.suptitle(f'DCR vs 50% rise time', horizontalalignment='center', fontsize=16)

        tlo, thi, tpb = 0, 800, 10

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
        # plt.savefig(f'./plots/normScan/cal_normScan/{runtype}_dcr_vs_tp0_50_run{run}.png', dpi=200)
        if runtype=='alp':
            plt.savefig(f'./plots/angleScan/{runtype}_DCRvTp050_{radius}mm_{angle_det}deg_run{run}.png', dpi=200)
        elif runtype=='bkg':
            plt.savefig(f'./plots/angleScan/{runtype}_DCRvTp050_run{run}.png', dpi=200)
        # plt.show()
        plt.clf()
        plt.close()

        # 1D AoE_________

        fig, ax = plt.subplots()
        fig.suptitle(f'A/E', horizontalalignment='center', fontsize=16)

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
        # plt.savefig(f'./plots/normScan/cal_normScan/{runtype}_1d_aoe_run{run}.png', dpi=200)
        if runtype=='alp':
            plt.savefig(f'./plots/angleScan/{runtype}_1dAoE_{radius}mm_{angle_det}deg_run{run}.png', dpi=200)
        elif runtype=='bkg':
            plt.savefig(f'./plots/angleScan/{runtype}_1dAoE_run{run}.png', dpi=200)
        plt.clf()
        plt.close()


        #-------------------------------------
        # Plots after alpha cuts
        #--------------------

#         if cut==False:
#             exit()
#         else:
#             continue

#         # Make calibrated energy spectrum_________

#         fig, ax = plt.subplots()
#         fig.suptitle(f'Energy after cut', horizontalalignment='center', fontsize=16)
#         elo, ehi, epb = 0, 6000, 10
#         # elo, ehi, epb = 0, 3000, 10
#         # elo, ehi, epb = 0, 6000, 10


#         nbx = int((ehi-elo)/epb)

#         energy_hist, bins = np.histogram(new_dcr_cut['trapEmax_cal'], bins=nbx,
#                 range=[elo, ehi])
#         energy_rt = np.divide(energy_hist, rt_min * 60)

#         plt.semilogy(bins[1:], energy_rt, ds='steps', c='b', lw=1) #, label=f'{etype}'

#         ax.set_xlabel('Energy (keV)', fontsize=16)
#         ax.set_ylabel('counts/sec', fontsize=16)
#         plt.setp(ax.get_xticklabels(), fontsize=14)
#         plt.setp(ax.get_yticklabels(), fontsize=14)

#         ax.text(0.95, 0.83, f'r = {radius} mm \ntheta = {angle_det} deg', verticalalignment='bottom',
#                     horizontalalignment='right', transform=ax.transAxes, color='green', fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

#         # plt.legend()
#         plt.title(f'\n{runtype} run {run}, {rt_min:.2f} mins', fontsize=12)
#         plt.tight_layout()
#         plt.savefig(f'./plots/normScan/cal_normScan/{runtype}_cut_energy_run{run}.png', dpi=200)
#         plt.clf()
#         plt.close()

#         # AoE vs E---------------------------------
#         fig, ax = plt.subplots()
#         alo, ahi, apb = 0.0, 0.09, 0.0001
#         if run>=60:
#             alo, ahi, apb = 0.005, 0.0905, 0.0001
#         elo, ehi, epb = 0, 6000, 10
#         # elo, ehi, epb = 0, 3000, 10
#         # elo, ehi, epb = 0, 6000, 10


#         nbx = int((ehi-elo)/epb)
#         nby = int((ahi-alo)/apb)

#         fig.suptitle(f'A/E vs Energy after cut', horizontalalignment='center', fontsize=16)

#         h = plt.hist2d(new_dcr_cut['trapEmax_cal'], new_dcr_cut['AoE'], bins=[nbx,nby],
#                     range=[[elo, ehi], [alo, ahi]], cmap='viridis', norm=LogNorm())

#         cb = plt.colorbar()
#         cb.set_label("counts", ha = 'right', va='center', rotation=270, fontsize=14)
#         cb.ax.tick_params(labelsize=12)
#         ax.set_xlabel('Energy (keV)', fontsize=16)
#         ax.set_ylabel('A/E (arb)', fontsize=16)
#         plt.setp(ax.get_xticklabels(), fontsize=14)
#         plt.setp(ax.get_yticklabels(), fontsize=14)

#         ax.text(0.95, 0.83, f'r = {radius} mm \ntheta = {angle_det} deg', verticalalignment='bottom',
#                     horizontalalignment='right', transform=ax.transAxes, color='green', fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

#         # plt.legend()
#         plt.title(f'\n{runtype} run {run}, {rt_min:.2f} mins', fontsize=12)
#         plt.tight_layout()
#         plt.savefig(f'./plots/normScan/cal_normScan/{runtype}_cut_AoE_run{run}.png', dpi=200)
#         # plt.show()

#         plt.clf()
#         plt.close()

#         # DCR vs E___________

#         fig, ax = plt.subplots()
#         dlo, dhi, dpb = -100, 300, 0.6
#         elo, ehi, epb = 0, 6000, 10
#         # elo, ehi, epb = 0, 3000, 10
#         # elo, ehi, epb = 0, 6000, 10

#         nbx = int((ehi-elo)/epb)
#         nby = int((dhi-dlo)/dpb)

#         fig.suptitle(f'DCR vs Energy after cut', horizontalalignment='center', fontsize=16)

#         h = plt.hist2d(new_dcr_cut['trapEmax_cal'], new_dcr_cut['dcr_linoff'], bins=[nbx,nby],
#                     range=[[elo, ehi], [dlo, dhi]], cmap='viridis', norm=LogNorm())

#         cb = plt.colorbar()
#         cb.set_label("counts", ha = 'right', va='center', rotation=270, fontsize=14)
#         cb.ax.tick_params(labelsize=12)
#         ax.set_xlabel('Energy (keV)', fontsize=16)
#         ax.set_ylabel('DCR (arb)', fontsize=16)
#         plt.setp(ax.get_xticklabels(), fontsize=14)
#         plt.setp(ax.get_yticklabels(), fontsize=14)

#         # plt.legend()

#         ax.text(0.95, 0.83, f'r = {radius} mm \ntheta = {angle_det} deg', verticalalignment='bottom',
#                     horizontalalignment='right', transform=ax.transAxes, color='green', fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

#         plt.title(f'\n{runtype} run {run}, {rt_min:.2f} mins', fontsize=12)
#         plt.tight_layout()
#         plt.savefig(f'./plots/normScan/cal_normScan/{runtype}_cut_dcr_run{run}.png', dpi=200)
#         # plt.show()
#         plt.clf()
#         plt.close()

#         # DCR vs A/E___________

#         fig, ax = plt.subplots()
#         dlo, dhi, dpb = -100, 300, 0.6
#         alo, ahi, apb = 0.0, 0.09, 0.0001
#         if run>=60:
#             alo, ahi, apb = 0.005, 0.0905, 0.0001

#         nbx = int((ahi-alo)/apb)
#         nby = int((dhi-dlo)/dpb)

#         fig.suptitle(f'A/E vs DCR after cut', horizontalalignment='center', fontsize=16)

#         h = plt.hist2d(new_dcr_cut['AoE'], new_dcr_cut['dcr_linoff'], bins=[nbx,nby],
#                     range=[[alo, ahi], [dlo, dhi]], cmap='viridis', norm=LogNorm())

#         cb = plt.colorbar()
#         cb.set_label("counts", ha = 'right', va='center', rotation=270, fontsize=14)
#         cb.ax.tick_params(labelsize=12)
#         ax.set_xlabel('A/E (arb)', fontsize=16)
#         ax.set_ylabel('DCR (arb)', fontsize=16)
#         plt.setp(ax.get_xticklabels(), fontsize=12)
#         plt.setp(ax.get_yticklabels(), fontsize=14)
#         # plt.legend()
#         ax.text(0.95, 0.83, f'r = {radius} mm \ntheta = {angle_det} deg', verticalalignment='bottom',
#                     horizontalalignment='right', transform=ax.transAxes, color='green', fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

#         plt.title(f'\n{runtype} run {run}, {rt_min:.2f} mins', fontsize=12)
#         plt.tight_layout()
#         plt.savefig(f'./plots/normScan/cal_normScan/{runtype}_cut_AoE_vs_dcr_run{run}.png', dpi=200)
#         # plt.show()
#         plt.clf()
#         plt.close()

#         # DCR vs tp_50___________

#         fig, ax = plt.subplots()
#         fig.suptitle(f'DCR vs 50% rise time after cut', horizontalalignment='center', fontsize=16)

#         dlo, dhi, dpb = -100, 200, 0.6
#         tlo, thi, tpb = 0, 700, 10

#         nbx = int((dhi-dlo)/dpb)
#         nby = int((thi-tlo)/tpb)

#         alpha_dcr_hist = plt.hist2d(new_dcr_cut['dcr_linoff'], new_dcr_cut['tp0_50'], bins=[nbx,nby],
#                 range=[[dlo, dhi], [tlo, thi]], cmap='viridis', norm=LogNorm())

#         cb = plt.colorbar()
#         cb.set_label("counts", ha = 'right', va='center', rotation=270, fontsize=14)
#         cb.ax.tick_params(labelsize=12)
#         ax.set_xlabel('DCR (arb)', fontsize=16)
#         ax.set_ylabel('tp 0-50 (ns)', fontsize=16)
#         plt.setp(ax.get_xticklabels(), fontsize=14)
#         plt.setp(ax.get_yticklabels(), fontsize=14)

#         # plt.legend()
#         ax.text(0.95, 0.83, f'r = {radius} mm \n theta = {angle_det} deg', verticalalignment='bottom',
#                     horizontalalignment='right', transform=ax.transAxes, color='green', fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

#         plt.title(f'\n{runtype} run {run}, {rt_min:.2f} mins', fontsize=12)
#         plt.tight_layout()
#         plt.savefig(f'./plots/normScan/cal_normScan/{runtype}_cut_dcr_vs_tp0_50_run{run}.png', dpi=200)
#         # plt.show()
#         plt.clf()
#         plt.close()

#         # 1D AoE_________

#         fig, ax = plt.subplots()
#         fig.suptitle(f'A/E after cut', horizontalalignment='center', fontsize=16)
#         alo, ahi, apb = 0.0, 0.09, 0.0001
#         if run>=60:
#             alo, ahi, apb = 0.005, 0.0905, 0.0001
#         nbx = int((ahi-alo)/apb)

#         aoe_hist, bins = np.histogram(new_dcr_cut['AoE'], bins=nbx,
#                 range=[alo, ahi])

#         plt.semilogy(bins[1:], aoe_hist, ds='steps', c='b', lw=1) #, label=f'{etype}'

#         ax.set_xlabel('A/E (arb)', fontsize=16)
#         ax.set_ylabel('counts', fontsize=16)
#         plt.setp(ax.get_xticklabels(), fontsize=14)
#         plt.setp(ax.get_yticklabels(), fontsize=14)

#         ax.text(0.95, 0.83, f'r = {radius} mm \ntheta = {angle_det} deg', verticalalignment='bottom',
#                     horizontalalignment='right', transform=ax.transAxes, color='green', fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

#         # plt.legend()
#         plt.title(f'\n{runtype} run {run}, {rt_min:.2f} mins', fontsize=12)
#         plt.tight_layout()
#         plt.savefig(f'./plots/normScan/cal_normScan/{runtype}_cut__1d_aoe_run{run}.png', dpi=200)
#         plt.clf()
#         plt.close()

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
