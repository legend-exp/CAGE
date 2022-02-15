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

import scipy.stats as stats

import pygama
from pygama import DataGroup
import pygama.lh5 as lh5
import pygama.analysis.histograms as pgh
import pygama.analysis.peak_fitting as pgf

import cage_utils

mpl.use('Agg')

def main():
    # runs = [64]
    runs = [60, 64, 66, 70, 72] # alpha runs for dsp_id = 2
    # runs = [62, 68, 74] #bkg runs for dsp_id = 2
    # campaign = 'angleScan/'
    campaign = 'new_normScan/'

    user = True
    hit = True
    cal = True
    etype = 'trapEftp' #use this even if you want calibrated-- `_cal` will get added to the end when `cal==True`

    # can specify which plots you actually want to make by adding them to `plot_list[]`
    plot_list = ['ToE_60']
    # plot_list = ['energy', 'energy_60', 'AoE', 'dcr', 'ToE', 'ToE_60', 'AoE_v_DCR', 'tp050_v_DCR', 'tp0220_v_DCR', 'ToE_v_DCR']
    # plot_list = ['energy', 'energy_60', 'AoE', 'dcr', 'ToE', 'AoE_v_DCR', 'tp050_v_DCR', 'tp0210_v_DCR', 'ToE_v_DCR']
    # plot_list = ['energy_60', 'ToE_60']



    # plot_dcr_slope(runs, corr_DCR=True, user=False, hit=True, cal=True, etype=etype, cut=True, campaign=campaign)

    # plot_energy(runs, etype=etype, corr_DCR=True, corr_AoE=True, user=True, hit=True, cal=True)
    
    normalized_dcr_AvE(runs, plot_list, corr_DCR=True, corr_AoE=True, corr_ToE=True, norm=True, user=user, hit=hit, cal=cal, etype=etype, cut=True, lowE=True, campaign=campaign)

def plot_dcr_slope(runs, corr_DCR=True, user=False, hit=True, cal=True, etype='trapEftp', cut=True, campaign=''):
    
    """
    This fits the DCR distribution over energy to a line, then plots the slope of that line.
    To compare how the slope of DCR changes between bkg and alp runs
    """

    if cal==True:
            #etype_cal = etype+'_cal'
            etype+='_cal'

    slopes = []
    slopes_err = []
    offsets = []
    offsets_err = []
#     runs = []

    for run in runs:
        print(run)

        df, runtype, rt_min, radius, angle_det, rotary = cage_utils.getDataFrame(run, user=user, hit=hit, cal=cal)
        



        # use baseline cut
        if run <79:
            bl_cut_lo, bl_cut_hi = 9150,9320
        if run>79 and run <117:
            bl_cut_lo, bl_cut_hi = 8500, 10000
        if run>=117:
            bl_cut_lo, bl_cut_hi = 9700, 9760

        df_cut = df.query(f'bl > {bl_cut_lo} and bl < {bl_cut_hi}').copy()

        const, offset, err = cage_utils.corrDCR(df_cut, etype, e_bins=300, elo=0, ehi=6000, dcr_fit_lo=-30, dcr_fit_hi=40)
        slopes.append(const)
        slopes_err.append(err[0])
        offsets.append(offset)
        offsets_err.append(err[1])
     # make plots with errorbars
#     print(const)
    fig, ax = plt.subplots()

#     plt.plot(runs, slopes, '.r')

    slope_plot = plt.errorbar(runs, slopes, yerr=slopes_err, marker = '.', ls='none', color = 'red', label='alpha runs')

    ax.set_xlabel('Run', fontsize=16)
    ax.set_ylabel('slope', fontsize=16)
    plt.setp(ax.get_xticklabels(), fontsize=14)
    plt.setp(ax.get_yticklabels(), fontsize=14)


#     plt.yscale('log')
    plt.title('DCR slope', fontsize=16)


    plt.savefig('./plots/new_normScan/alpha_dcr_slope.png', dpi=200)
    plt.clf()
    plt.close()


    offset_plot = plt.errorbar(runs, offsets, yerr=offsets_err, marker = '.', ls='none', color = 'red', label='alpha runs')

    ax.set_xlabel('Run', fontsize=16)
    ax.set_ylabel('slope', fontsize=16)
    plt.setp(ax.get_xticklabels(), fontsize=14)
    plt.setp(ax.get_yticklabels(), fontsize=14)


#     plt.yscale('log')
    plt.title('DCR offset', fontsize=16)


    plt.savefig('./plots/new_normScan/alpha_dcr_offset.png', dpi=200)
    plt.clf()
    plt.close()


def normalized_dcr_AvE(runs, plot_list=[], corr_DCR=True, corr_AoE=True, corr_ToE=True, norm=True, user=False, hit=True, cal=True, etype='trapEftp', cut=True, lowE=False, cut_str = '', campaign=''):
    
    """
    Create as many or few plots as you want by specifying what you want to plot in `plot_list[]`
    Can be normalized by runtime with `norm==True`
    
    `cut==True` applies data-cleaning cuts, which must have been defined earlier 
    (by running dataCleaning.ipynb to write cut values to cuts.json)
    
    corr_DCR corrects the slope of DCR to make it flat over energy and shifts the centroid to 0
    
    corr_ToE (corr_AoE) shifts the T/E (A/E) distribution so that the mode is at 0 for easier comparison
    """

    if cal==True:
            #etype_cal = etype+'_cal'
            etype+='_cal'

    for run in runs:


        df, dg, runtype, rt_min, radius, angle_det, rotary = cage_utils.getDataFrame(run, user=user, hit=hit,
                                                                                     cal=cal, lowE=False)
        radius_str = str(radius)
        radius_fn = radius_str.replace('.', '_')


        # use baseline cut
        # Joule recommends using dataCleaning.ipynb to write cut values to cuts.json
        # then apply cuts with cage_utils.apply_DC_Cuts
        # if run <79:
            # bl_cut_lo, bl_cut_hi = 9150,9320
        # if run>79 and run <117:
            # bl_cut_lo, bl_cut_hi = 8500, 10000
        # if run>=117:
            # bl_cut_lo, bl_cut_hi = 9700, 9760

        # df = df_raw.query(f'bl > {bl_cut_lo} and bl < {bl_cut_hi}').copy()
        
        
        # corr_DCR corrects the slope of DCR and makes the centroid at 0
        if corr_DCR==True and run>57:
            const, offset, err = cage_utils.corrDCR(df, etype, e_bins=300, elo=0, ehi=6000, dcr_fit_lo=-30,
                                                    dcr_fit_hi=40)
            df['dcr_plot'] = df['dcr']-offset + ((-1*const))*df[etype]
        elif corr_DCR==True and run<57:
            const = const = 0.0011
            df['dcr_plot'] = df['dcr'] - const*df['trapEftp']
        else:
            df['dcr_plot'] = df['dcr']

        # corr_AoE shifts the A/E distribution so that the mod is at 0 for easier comparison
        if corr_AoE==True:
            AoE_mode = cage_utils.mode_hist(df, param='AoE', a_bins=1000, alo=0.005, ahi=0.075, cut=False,
                                            cut_str='')
            df['AoE_plot'] = df['AoE'] - AoE_mode
        else:
            df['AoE_plot'] = df['AoE']

        # corr_ToE shifts the T/E distribution so that the mod is at 0 for easier comparison
        if corr_ToE==True:
            ToE_mode = cage_utils.mode_hist(df, param='ToE', a_bins=1000, alo=0.30, ahi=0.45, cut=False, cut_str='')
            df['ToE_plot'] = df['ToE'] - ToE_mode
        else:
            df['ToE_plot'] = df['ToE']


        #create timepoints 0-50 and 02-10, which are helpful for discriminating alphas
        df['tp0_50'] = df['tp_50']- df['tp_0']
        df['tp0210'] = df['tp_10'] - df['tp_02']
        df['tp0220'] = df['tp_20'] - df['tp_02']

        # create cut if relevant
        if cut == True:
            print(f'Applying data-cleaning cuts')
            df_cut = cage_utils.apply_DC_Cuts(run, df)
        else:
            df_cut = df

        if norm==True:
            rt = np.array([(1/rt_min)])
            wts = np.repeat(rt, len(df_cut[etype]))
        else:
            rt = np.array([(1/1.)])
            wts = np.repeat(rt, len(df_cut[etype]))

        # select energy type and energy range
        if cal==False:
            elo, ehi, epb = 0, 10000, 10 #entire enerty range trapEftp
            e_unit = ' (uncal)'
        elif cal==True:
            elo, ehi, epb = 0, 6000, 10.
            elo_60, ehi_60, epb_60 = 40, 80, 1.
            # etype=etype_cal
            e_unit = ' (keV)'

        #-------------------------------------
        # Plots before alpha cuts
        #--------------------

        if 'energy' in plot_list:

            # Make (calibrated) energy spectrum_________

            fig, ax = plt.subplots()
            fig.suptitle(f'Energy', horizontalalignment='center', fontsize=16, y=0.95)

            nbx = int((ehi-elo)/epb)

            energy_hist_norm, bins = np.histogram(df_cut[etype], bins=nbx,
                                            range=[elo, ehi], weights=wts)

            plt.semilogy(bins[1:], energy_hist_norm, ds='steps', c='b', lw=1) #, label=f'{etype}'

            ax.set_xlabel(f'Energy{e_unit}', fontsize=16)
            if norm==True:
                ax.set_ylabel('counts / (min $\cdot$ 10 keV)', fontsize=16)
            else:
                ax.set_ylabel('counts', fontsize=16)
            plt.ylim(0.01,200)
            plt.xlim(10., ehi)
            plt.setp(ax.get_xticklabels(), fontsize=14)
            plt.setp(ax.get_yticklabels(), fontsize=14)
            
            ax.text(0.95, 0.81, f'r = {radius} mm', verticalalignment='bottom',
                        horizontalalignment='right', transform=ax.transAxes, color='green', fontsize=14,
                        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

            # ax.text(0.95, 0.81, f'r = {radius} mm \ntheta = {angle_det} deg', verticalalignment='bottom',
                        # horizontalalignment='right', transform=ax.transAxes, color='green', fontsize=14,
                        # bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

            # plt.legend()
            plt.title(f'{runtype} run {run}, {rt_min:.2f} mins', fontsize=12)
            plt.tight_layout()
            # plt.savefig(f'./plots/normScan/cal_normScan/{runtype}_energy_run{run}.png', dpi=200)
            if runtype=='alp' and norm==True:
                plt.savefig(f'./plots/{campaign}normalized_{runtype}_energy_{radius_fn}mm_{angle_det}deg_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}normalized_{runtype}_energy_{radius_fn}mm_{angle_det}deg_run{run}.pdf', dpi=200)
            elif runtype=='bkg' and norm==True:
                plt.savefig(f'./plots/{campaign}normalized_{runtype}_energy_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}normalized_{runtype}_energy_run{run}.pdf', dpi=200)
            elif runtype=='alp' and norm==False:
                plt.savefig(f'./plots/{campaign}{runtype}_energy_{radius_fn}mm_{angle_det}deg_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}{runtype}_energy_{radius_fn}mm_{angle_det}deg_run{run}.pdf', dpi=200)
            elif runtype=='bkg' and norm==False:
                plt.savefig(f'./plots/{campaign}{runtype}_energy_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}{runtype}_energy_run{run}.pdf', dpi=200)
                
            plt.clf()
            plt.close()

        if 'energy_60' in plot_list:

            fig, ax = plt.subplots()
            fig.suptitle(f'Energy', horizontalalignment='center', fontsize=16, y=0.95)
            
            nbx_60 = int((ehi_60-elo_60)/epb_60)
                

            energy_hist_norm_60, bins_60 = np.histogram(df_cut[etype], bins=nbx_60,
                                            range=[elo_60, ehi_60], weights=wts)

            plt.plot(bins_60[1:], energy_hist_norm_60, ds='steps', c='b', lw=1) #, label=f'{etype}'

            ax.set_xlabel(f'Energy{e_unit}', fontsize=16)
            if norm==True:
                ax.set_ylabel('counts / (min $\cdot$ keV)', fontsize=16)
            else:
                ax.set_ylabel('counts', fontsize=16)
            plt.setp(ax.get_xticklabels(), fontsize=14)
            plt.setp(ax.get_yticklabels(), fontsize=14)
            
            plt.ylim(3, 10)

            ax.text(0.95, 0.81, f'r = {radius} mm', verticalalignment='bottom',
                        horizontalalignment='right', transform=ax.transAxes, color='green', fontsize=14,
                        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

            # plt.legend()
            plt.title(f'{runtype} run {run}, {rt_min:.2f} mins', fontsize=12)
            plt.tight_layout()

            if runtype=='alp' and norm==True:
                    plt.savefig(f'./plots/{campaign}normalized_{runtype}_energy_60keV_{radius_fn}mm_{angle_det}deg_run{run}.png', dpi=200)
                    plt.savefig(f'./plots/{campaign}normalized_{runtype}_energy_60keV_{radius_fn}mm_{angle_det}deg_run{run}.pdf', dpi=200)
            elif runtype=='bkg' and norm==True:
                plt.savefig(f'./plots/{campaign}normalized_{runtype}_energy_60keV_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}normalized_{runtype}_energy_60keV_run{run}.pdf', dpi=200)
                    
            elif runtype=='alp' and norm==False:
                plt.savefig(f'./plots/{campaign}{runtype}_energy_60keV_{radius_fn}mm_{angle_det}deg_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}{runtype}_energy_60keV_{radius_fn}mm_{angle_det}deg_run{run}.pdf', dpi=200)
            elif runtype=='bkg' and norm==False:
                plt.savefig(f'./plots/{campaign}{runtype}_energy_60keV_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}{runtype}_energy_60keV_run{run}.pdf', dpi=200)


            plt.clf()
            plt.close()

        # AoE vs E---------------------------------
        if 'AoE' in plot_list:

            # normalized by runtime
            fig, ax = plt.subplots()
            alo, ahi, apb = 0.0, 0.09, 0.0001
            if run>=36:
                alo, ahi, apb = 0.005, 0.075, 0.0001
            if run>=117:
                alo, ahi, apb = 0.0, 0.15, 0.00015

            if corr_AoE==True:
                alo, ahi, apb= -0.03, 0.03, 0.0005

            nbx = int((ehi-elo)/epb)
            nby = int((ahi-alo)/apb)

            fig.suptitle(f'A/E vs Energy', ha='center', va='top', fontsize=16, y=0.95)

            aoe_hist_norm, xedges, yedges = np.histogram2d(df_cut[etype], df_cut['AoE_plot'], bins=[nbx, nby], range=([elo, ehi], [alo, ahi]), weights=wts)
            X, Y = np.mgrid[elo:ehi:nbx*1j, alo:ahi:nby*1j]

#             aoe_hist_norm = np.divide(aoe_hist, (rt_min))

            pcm = plt.pcolormesh(X, Y, aoe_hist_norm, shading='nearest', norm=LogNorm(0.001, 10)) #0.002, 0.2

            cb = plt.colorbar()
            if norm==True:
                cb.set_label("counts/min", ha = 'right', va='center', rotation=270, fontsize=14)
            else:
                cb.set_label("counts", ha = 'right', va='center', rotation=270, fontsize=14)
            cb.ax.tick_params(labelsize=12)
            ax.set_xlabel(f'Energy {e_unit}', fontsize=16)
            ax.set_ylabel('A/E (arb)', fontsize=16)
            plt.setp(ax.get_xticklabels(), fontsize=14)
            plt.setp(ax.get_yticklabels(), fontsize=14)


            ax.text(0.95, 0.81, f'r = {radius} mm', verticalalignment='bottom',
                        horizontalalignment='right', transform=ax.transAxes, color='green', fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

            # plt.legend()
            plt.title(f'{runtype} run {run}, {rt_min:.2f} mins', fontsize=12)
            plt.tight_layout()
            # plt.savefig(f'./plots/normScan/cal_normScan/{runtype}_AoE_run{run}.png', dpi=200)
            if runtype=='alp' and norm==True:
                plt.savefig(f'./plots/{campaign}normalized_{runtype}_AoE_{radius_fn}mm_{angle_det}deg_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}normalized_{runtype}_AoE_{radius_fn}mm_{angle_det}deg_run{run}.pdf', dpi=200)
            elif runtype=='bkg' and norm==True:
                plt.savefig(f'./plots/{campaign}normalized_{runtype}_AoE_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}normalized_{runtype}_AoE_run{run}.pdf', dpi=200)
            elif runtype=='alp' and norm==False:
                plt.savefig(f'./plots/{campaign}{runtype}_AoE_{radius_fn}mm_{angle_det}deg_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}{runtype}_AoE_{radius_fn}mm_{angle_det}deg_run{run}.pdf', dpi=200)
            elif runtype=='bkg' and norm==False:
                plt.savefig(f'./plots/{campaign}{runtype}_AoE_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}{runtype}_AoE_run{run}.pdf', dpi=200)
                # plt.show()


            plt.clf()
            plt.close()
        # AoE vs E---------------------------------
        if 'ToE' in plot_list:

            # normalized by runtime
            fig, ax = plt.subplots()
            ToElo, ToEhi, ToEpb = 0.0, 0.5, 0.001

            if corr_ToE==True:
                ToElo, ToEhi, ToEpb= -0.3, 0.3, 0.005

            nbx = int((ehi-elo)/epb)
            nby = int((ToEhi-ToElo)/ToEpb)

            fig.suptitle(f'T/E vs Energy', horizontalalignment='center', fontsize=16, y=0.95)

            ToE_hist_norm, xedges, yedges = np.histogram2d(df_cut[etype], df_cut['ToE_plot'], bins=[nbx, nby], range=([elo, ehi], [ToElo, ToEhi]), weights=wts)
            X, Y = np.mgrid[elo:ehi:nbx*1j, ToElo:ToEhi:nby*1j]

#             aoe_hist_norm = np.divide(aoe_hist, (rt_min))

            pcm = plt.pcolormesh(X, Y, ToE_hist_norm, shading='nearest', norm=LogNorm(0.001, 10)) #0.002, 0.2

            cb = plt.colorbar()
            if norm==True:
                cb.set_label("counts/min", ha = 'right', va='center', rotation=270, fontsize=14)
            else:
                cb.set_label("counts", ha = 'right', va='center', rotation=270, fontsize=14)
            cb.ax.tick_params(labelsize=12)
            ax.set_xlabel(f'Energy {e_unit}', fontsize=16)
            ax.set_ylabel('T/E (arb)', fontsize=16)
            plt.setp(ax.get_xticklabels(), fontsize=14)
            plt.setp(ax.get_yticklabels(), fontsize=14)


            ax.text(0.95, 0.81, f'r = {radius} mm', verticalalignment='bottom',
                        horizontalalignment='right', transform=ax.transAxes, color='green', fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

            # plt.legend()
            plt.title(f'{runtype} run {run}, {rt_min:.2f} mins', fontsize=12)
            plt.tight_layout()
            # plt.savefig(f'./plots/normScan/cal_normScan/{runtype}_AoE_run{run}.png', dpi=200)
            if runtype=='alp' and norm==True:
                plt.savefig(f'./plots/{campaign}normalized_{runtype}_ToE_{radius_fn}mm_{angle_det}deg_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}normalized_{runtype}_ToE_{radius_fn}mm_{angle_det}deg_run{run}.pdf', dpi=200)
            elif runtype=='bkg' and norm==True:
                plt.savefig(f'./plots/{campaign}normalized_{runtype}_ToE_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}normalized_{runtype}_ToE_run{run}.pdf', dpi=200)
            elif runtype=='alp' and norm==False:
                plt.savefig(f'./plots/{campaign}{runtype}_ToE_{radius_fn}mm_{angle_det}deg_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}{runtype}_ToE_{radius_fn}mm_{angle_det}deg_run{run}.pdf', dpi=200)
            elif runtype=='bkg' and norm==False:
                plt.savefig(f'./plots/{campaign}{runtype}_ToE_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}{runtype}_ToE_run{run}.pdf', dpi=200)
                # plt.show()
            plt.clf()
            plt.close()
                
        if 'ToE_60' in plot_list:

            # normalized by runtime
            fig, ax = plt.subplots()
            ToElo, ToEhi, ToEpb = 0.0, 0.5, 0.001

            if corr_ToE==True:
                ToElo, ToEhi, ToEpb= -0.1, 0.1, 0.005

            nbx_60 = int((ehi_60-elo_60)/epb_60)
            nby = int((ToEhi-ToElo)/ToEpb)

            fig.suptitle(f'T/E vs Energy', horizontalalignment='center', fontsize=16, y=0.95)

            ToE_hist_norm, xedges, yedges = np.histogram2d(df_cut[etype], df_cut['ToE_plot'], bins=[nbx_60, nby], range=([elo_60, ehi_60], [ToElo, ToEhi]), weights=wts)
            X, Y = np.mgrid[elo_60:ehi_60:nbx_60*1j, ToElo:ToEhi:nby*1j]

#             aoe_hist_norm = np.divide(aoe_hist, (rt_min))

            pcm = plt.pcolormesh(X, Y, ToE_hist_norm, shading='nearest', vmin=0.0, vmax=1.1) #, norm=LogNorm(0.001, 1)#0.002, 0.2

            cb = plt.colorbar()
            if norm==True:
                cb.set_label("counts/min", ha = 'right', va='center', rotation=270, fontsize=14)
            else:
                cb.set_label("counts", ha = 'right', va='center', rotation=270, fontsize=14)
            cb.ax.tick_params(labelsize=12)
            ax.set_xlabel(f'Energy {e_unit}', fontsize=16)
            ax.set_ylabel('T/E (arb)', fontsize=16)
            plt.setp(ax.get_xticklabels(), fontsize=14)
            plt.setp(ax.get_yticklabels(), fontsize=14)


            ax.text(0.95, 0.81, f'r = {radius} mm', verticalalignment='bottom',
                        horizontalalignment='right', transform=ax.transAxes, color='green', fontsize=14, bbox={'facecolor': 'white', 'alpha': 1., 'pad': 10})

            # plt.legend()
            plt.title(f'{runtype} run {run}, {rt_min:.2f} mins', fontsize=12)
            plt.tight_layout()

            if runtype=='alp' and norm==True:
                plt.savefig(f'./plots/{campaign}normalized_{runtype}_ToE_60keV_{radius_fn}mm_{angle_det}deg_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}normalized_{runtype}_ToE_60keV_{radius_fn}mm_{angle_det}deg_run{run}.pdf', dpi=200)
            elif runtype=='bkg' and norm==True:
                plt.savefig(f'./plots/{campaign}normalized_{runtype}_ToE_60keV_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}normalized_{runtype}_ToE_60keV_run{run}.pdf', dpi=200)
                    
            elif runtype=='alp' and norm==False:
                plt.savefig(f'./plots/{campaign}{runtype}_ToE_60keV_{radius_fn}mm_{angle_det}deg_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}{runtype}_ToE_60keV_{radius_fn}mm_{angle_det}deg_run{run}.pdf', dpi=200)
            elif runtype=='bkg' and norm==False:
                plt.savefig(f'./plots/{campaign}{runtype}_ToE_60keV_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}{runtype}_ToE_60keV_run{run}.pdf', dpi=200)

            plt.clf()
            plt.close()

        # DCR vs E___________
        # create new new DCR
        if 'dcr' in plot_list:

            fig, ax = plt.subplots()



            if run>=36 and run<117:
                dlo, dhi = -40, 170
                d_bins = 200
            elif run>=117:
                # dlo, dhi, dpb = -20., 40, 0.1
                dlo, dhi = -40, 170
                d_bins = 200

            elo_dcr, ehi_dcr, epb_dcr = 50, 6000, 10

            dcr_nbx = int((ehi_dcr-elo_dcr)/epb_dcr)

            fig.suptitle(f'DCR vs Energy', horizontalalignment='center', fontsize=16, y=0.95)

            dcr_hist_norm, xedges, yedges = np.histogram2d(df_cut[etype], df_cut['dcr_plot'], bins=[dcr_nbx, d_bins], range=([elo_dcr, ehi_dcr], [dlo, dhi]), weights=wts)
            X, Y = np.mgrid[elo_dcr:ehi_dcr:dcr_nbx*1j, dlo:dhi:d_bins*1j]

            # dcr_hist_norm = np.divide(dcr_hist, (rt_min))

            pcm = plt.pcolormesh(X, Y, dcr_hist_norm, shading='nearest', norm=LogNorm(0.001, 10))

            cb = plt.colorbar()
            if norm==True:
                cb.set_label("counts/min", ha = 'right', va='center', rotation=270, fontsize=14)
            else:
                cb.set_label("counts", ha = 'right', va='center', rotation=270, fontsize=14)
            cb.ax.tick_params(labelsize=12)
            ax.set_xlabel('Energy (keV)', fontsize=16)
            ax.set_ylabel('DCR (arb)', fontsize=16)
            plt.setp(ax.get_xticklabels(), fontsize=14)
            plt.setp(ax.get_yticklabels(), fontsize=14)

            # plt.legend()

            ax.text(0.95, 0.81, f'r = {radius} mm', verticalalignment='bottom',
                    horizontalalignment='right', transform=ax.transAxes, color='green', fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

            plt.title(f'{runtype} run {run}, {rt_min:.2f} mins', fontsize=12)
            plt.tight_layout()
            # plt.savefig(f'./plots/normScan/cal_normScan/{runtype}_dcr_run{run}.png', dpi=200)
            if runtype=='alp' and norm==True:
                plt.savefig(f'./plots/{campaign}normalized_{runtype}_DCR_{radius_fn}mm_{angle_det}deg_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}normalized_{runtype}_DCR_{radius_fn}mm_{angle_det}deg_run{run}.pdf', dpi=200)
            elif runtype=='bkg' and norm==True:
                plt.savefig(f'./plots/{campaign}normalized_{runtype}_DCR_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}normalized_{runtype}_DCR_run{run}.pdf', dpi=200)
            elif runtype=='alp' and norm==False:
                plt.savefig(f'./plots/{campaign}{runtype}_DCR_{radius_fn}mm_{angle_det}deg_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}{runtype}_DCR_{radius_fn}mm_{angle_det}deg_run{run}.pdf', dpi=200)
            elif runtype=='bkg' and norm==False:
                plt.savefig(f'./plots/{campaign}{runtype}_DCR_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}{runtype}_DCR_run{run}.pdf', dpi=200)
                
            plt.xlim(40, 80)
            if runtype=='alp' and norm==True:
                plt.savefig(f'./plots/{campaign}normalized_{runtype}_DCR_60keV_{radius_fn}mm_{angle_det}deg_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}normalized_{runtype}_DCR_60keV_{radius_fn}mm_{angle_det}deg_run{run}.pdf', dpi=200)
            elif runtype=='bkg' and norm==True:
                plt.savefig(f'./plots/{campaign}normalized_{runtype}_DCR_60keV_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}normalized_{runtype}_DCR_60keV_run{run}.pdf', dpi=200)
            elif runtype=='alp' and norm==False:
                plt.savefig(f'./plots/{campaign}{runtype}_DCR_60keV_{radius_fn}mm_{angle_det}deg_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}{runtype}_DCR_60keV_{radius_fn}mm_{angle_det}deg_run{run}.pdf', dpi=200)
            elif runtype=='bkg' and norm==False:
                plt.savefig(f'./plots/{campaign}{runtype}_DCR_60keV_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}{runtype}_DCR_60keV_run{run}.pdf', dpi=200)
            # plt.show()
            plt.clf()
            plt.close()

        # DCR vs A/E___________
        if 'AoE_v_DCR' in plot_list:

            fig, ax = plt.subplots()

            if run>=36 and run<117:
                dlo, dhi = -40, 170
                d_bins = 200
            elif run>=117:
                # dlo, dhi, dpb = -20., 40, 0.1
                dlo, dhi = -40, 170
                d_bins = 200

            nbx = int((ahi-alo)/apb)
            #nby = int((dhi-dlo)/dpb)

            fig.suptitle(f'A/E vs DCR', horizontalalignment='center', fontsize=16, y=0.95)

            aoeVdcr_hist_norm, xedges, yedges = np.histogram2d(df_cut['AoE_plot'], df_cut['dcr_plot'], bins=[nbx, d_bins], range=([alo, ahi], [dlo, dhi]), weights=wts)
            X, Y = np.mgrid[alo:ahi:nbx*1j, dlo:dhi:d_bins*1j]

            #aoeVdcr_hist_norm = np.divide(aoeVdcr_hist, (rt_min))

            pcm = plt.pcolormesh(X, Y, aoeVdcr_hist_norm, shading='nearest', norm=LogNorm(0.001, 10))

            cb = plt.colorbar()
            if norm==True:
                cb.set_label("counts/min", ha = 'right', va='center', rotation=270, fontsize=14)
            else:
                cb.set_label("counts", ha = 'right', va='center', rotation=270, fontsize=14)
            cb.ax.tick_params(labelsize=12)
            ax.set_xlabel('A/E (arb)', fontsize=16)
            ax.set_ylabel('DCR (arb)', fontsize=16)
            plt.setp(ax.get_xticklabels(), fontsize=12)
            plt.setp(ax.get_yticklabels(), fontsize=14)

            # plt.legend()
            ax.text(0.95, 0.81, f'r = {radius} mm', verticalalignment='bottom',
                        horizontalalignment='right', transform=ax.transAxes, color='green', fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

            plt.title(f'{runtype} run {run}, {rt_min:.2f} mins', fontsize=12)
            plt.tight_layout()
            # plt.savefig(f'./plots/normScan/cal_normScan/{runtype}_AoE_vs_dcr_run{run}.png', dpi=200)
            if runtype=='alp' and norm==True:
                plt.savefig(f'./plots/{campaign}normalized_{runtype}_AoEvDCR_{radius_fn}mm_{angle_det}deg_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}normalized_{runtype}_AoEvDCR_{radius_fn}mm_{angle_det}deg_run{run}.pdf', dpi=200)
            elif runtype=='bkg' and norm==True:
                plt.savefig(f'./plots/{campaign}normalized_{runtype}_AoEvDCR_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}normalized_{runtype}_AoEvDCR_run{run}.pdf', dpi=200)
            elif runtype=='alp' and norm==False:
                plt.savefig(f'./plots/{campaign}{runtype}_AoEvDCR_{radius_fn}mm_{angle_det}deg_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}{runtype}_AoEvDCR_{radius_fn}mm_{angle_det}deg_run{run}.pdf', dpi=200)
            elif runtype=='bkg' and norm==False:
                plt.savefig(f'./plots/{campaign}{runtype}_AoEvDCR_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}{runtype}_AoEvDCR_run{run}.pdf', dpi=200)
            # plt.show()
            plt.clf()
            plt.close()

        # DCR vs T/E___________
        if 'ToE_v_DCR' in plot_list:

            fig, ax = plt.subplots()

            if run>=36 and run<117:
                dlo, dhi = -40, 170
                d_bins = 200
            elif run>=117:
                # dlo, dhi, dpb = -20., 40, 0.1
                dlo, dhi = -40, 170
                d_bins = 200

            ToElo, ToEhi, ToEpb = 0.0, 0.5, 0.001

            if corr_ToE==True:
                ToElo, ToEhi, ToEpb= -0.2, 0.2, 0.005

            nbx = int((ToEhi-ToElo)/ToEpb)


            fig.suptitle(f'T/E vs DCR', horizontalalignment='center', fontsize=16, y=0.95)

            ToEVdcr_hist_norm, xedges, yedges = np.histogram2d(df_cut['ToE_plot'], df_cut['dcr_plot'], bins=[nbx, d_bins], range=([ToElo, ToEhi], [dlo, dhi]), weights=wts)
            X, Y = np.mgrid[ToElo:ToEhi:nbx*1j, dlo:dhi:d_bins*1j]

            #aoeVdcr_hist_norm = np.divide(aoeVdcr_hist, (rt_min))

            pcm = plt.pcolormesh(X, Y, ToEVdcr_hist_norm, shading='nearest', norm=LogNorm(0.001, 10))

            cb = plt.colorbar()
            if norm==True:
                cb.set_label("counts/min", ha = 'right', va='center', rotation=270, fontsize=14)
            else:
                cb.set_label("counts", ha = 'right', va='center', rotation=270, fontsize=14)
            cb.ax.tick_params(labelsize=12)
            ax.set_xlabel('T/E (arb)', fontsize=16)
            ax.set_ylabel('DCR (arb)', fontsize=16)
            plt.setp(ax.get_xticklabels(), fontsize=12)
            plt.setp(ax.get_yticklabels(), fontsize=14)

            # plt.legend()
            ax.text(0.95, 0.81, f'r = {radius} mm', verticalalignment='bottom',
                        horizontalalignment='right', transform=ax.transAxes, color='green', fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

            plt.title(f'{runtype} run {run}, {rt_min:.2f} mins', fontsize=12)
            plt.tight_layout()
            # plt.savefig(f'./plots/normScan/cal_normScan/{runtype}_AoE_vs_dcr_run{run}.png', dpi=200)
            if runtype=='alp' and norm==True:
                plt.savefig(f'./plots/{campaign}normalized_{runtype}_ToEvDCR_{radius_fn}mm_{angle_det}deg_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}normalized_{runtype}_ToEvDCR_{radius_fn}mm_{angle_det}deg_run{run}.pdf', dpi=200)
            elif runtype=='bkg' and norm==True:
                plt.savefig(f'./plots/{campaign}normalized_{runtype}_ToEvDCR_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}normalized_{runtype}_ToEvDCR_run{run}.pdf', dpi=200)
            elif runtype=='alp' and norm==False:
                plt.savefig(f'./plots/{campaign}{runtype}_ToEvDCR_{radius_fn}mm_{angle_det}deg_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}{runtype}_ToEvDCR_{radius_fn}mm_{angle_det}deg_run{run}.pdf', dpi=200)
            elif runtype=='bkg' and norm==False:
                plt.savefig(f'./plots/{campaign}{runtype}_ToEvDCR_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}{runtype}_ToEvDCR_run{run}.pdf', dpi=200)
            # plt.show()
            plt.clf()
            plt.close()

        # DCR vs tp_50___________
        if 'tp050_v_DCR' in plot_list:

            fig, ax = plt.subplots()
            fig.suptitle(f'DCR vs 50% rise time', horizontalalignment='center', fontsize=16, y=0.95)

            tlo, thi, tpb = 0, 400, 10

            if run>=36 and run<117:
                dlo, dhi = -40, 170
                d_bins = 200
            elif run>=117:
                # dlo, dhi, dpb = -20., 40, 0.1
                dlo, dhi = -40, 170
                d_bins = 200

            nby = int((thi-tlo)/tpb)

            DCRvTp050_hist, xedges, yedges = np.histogram2d(df_cut['dcr_plot'], df_cut['tp0_50'], bins=[d_bins, nby], range=([dlo, dhi], [tlo, thi]))
            X, Y = np.mgrid[dlo:dhi:d_bins*1j, tlo:thi:nby*1j]

            DCRvTp050_hist_norm = np.divide(DCRvTp050_hist, (rt_min))

            pcm = plt.pcolormesh(X, Y, DCRvTp050_hist_norm, shading='nearest', norm=LogNorm(0.001, 10))

            cb = plt.colorbar()
            if norm==True:
                cb.set_label("counts/min", ha = 'right', va='center', rotation=270, fontsize=14)
            else:
                cb.set_label("counts", ha = 'right', va='center', rotation=270, fontsize=14)
            cb.ax.tick_params(labelsize=12)
            ax.set_xlabel('DCR (arb)', fontsize=16)
            ax.set_ylabel('tp 0-50 (ns)', fontsize=16)
            plt.setp(ax.get_xticklabels(), fontsize=14)
            plt.setp(ax.get_yticklabels(), fontsize=14)

            # plt.legend()
            ax.text(0.95, 0.81, f'r = {radius} mm', verticalalignment='bottom',
                        horizontalalignment='right', transform=ax.transAxes, color='green', fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.95, 'pad': 10})

            plt.title(f'{runtype} run {run}, {rt_min:.2f} mins', fontsize=12)
            plt.tight_layout()
            # plt.savefig(f'./plots/normScan/cal_normScan/{runtype}_dcr_vs_tp0_50_run{run}.png', dpi=200)
            if runtype=='alp' and norm==True:
                plt.savefig(f'./plots/{campaign}normalized_{runtype}_DCRvTp050_{radius_fn}mm_{angle_det}deg_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}normalized_{runtype}_DCRvTp050_{radius_fn}mm_{angle_det}deg_run{run}.pdf', dpi=200)
            elif runtype=='bkg' and norm==True:
                plt.savefig(f'./plots/{campaign}normalized_{runtype}_DCRvTp050_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}normalized_{runtype}_DCRvTp050_run{run}.pdf', dpi=200)
            elif runtype=='alp' and norm==False:
                plt.savefig(f'./plots/{campaign}{runtype}_DCRvTp050_{radius_fn}mm_{angle_det}deg_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}{runtype}_DCRvTp050_{radius_fn}mm_{angle_det}deg_run{run}.pdf', dpi=200)
            elif runtype=='bkg' and norm==False:
                plt.savefig(f'./plots/{campaign}{runtype}_DCRvTp050_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}{runtype}_DCRvTp050_run{run}.pdf', dpi=200)
            # plt.show()
            plt.clf()
            plt.close()
            
        # DCR vs tp_0210___________
        if 'tp0210_v_DCR' in plot_list:

            fig, ax = plt.subplots()
            fig.suptitle(f'DCR vs 2-10% rise time', horizontalalignment='center', fontsize=16, y=0.95)

            tlo, thi, tpb = 0, 500, 10

            if run>=36 and run<117:
                dlo, dhi = -40, 170
                d_bins = 200
            elif run>=117:
                # dlo, dhi, dpb = -20., 40, 0.1
                dlo, dhi = -40, 170
                d_bins = 200

            nby = int((thi-tlo)/tpb)

            DCRvTp050_hist, xedges, yedges = np.histogram2d(df_cut['dcr_plot'], df_cut['tp0210'], bins=[d_bins, nby], range=([dlo, dhi], [tlo, thi]))
            X, Y = np.mgrid[dlo:dhi:d_bins*1j, tlo:thi:nby*1j]

            DCRvTp0210_hist_norm = np.divide(DCRvTp050_hist, (rt_min))

            pcm = plt.pcolormesh(X, Y, DCRvTp0210_hist_norm, shading='nearest', norm=LogNorm(0.001, 10))

            cb = plt.colorbar()
            if norm==True:
                cb.set_label("counts/min", ha = 'right', va='center', rotation=270, fontsize=14)
            else:
                cb.set_label("counts", ha = 'right', va='center', rotation=270, fontsize=14)
            cb.ax.tick_params(labelsize=12)
            ax.set_xlabel('DCR (arb)', fontsize=16)
            ax.set_ylabel('tp 02-10 (ns)', fontsize=16)
            plt.setp(ax.get_xticklabels(), fontsize=14)
            plt.setp(ax.get_yticklabels(), fontsize=14)

            # plt.legend()
            ax.text(0.95, 0.81, f'r = {radius} mm', verticalalignment='bottom',
                        horizontalalignment='right', transform=ax.transAxes, color='green', fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.95, 'pad': 10})

            plt.title(f'{runtype} run {run}, {rt_min:.2f} mins', fontsize=12)
            plt.tight_layout()
            # plt.savefig(f'./plots/normScan/cal_normScan/{runtype}_dcr_vs_tp0_50_run{run}.png', dpi=200)
            if runtype=='alp' and norm==True:
                    plt.savefig(f'./plots/{campaign}normalized_{runtype}_DCRvTp0210_{radius_fn}mm_{angle_det}deg_run{run}.png', dpi=200)
                    plt.savefig(f'./plots/{campaign}normalized_{runtype}_DCRvTp0210_{radius_fn}mm_{angle_det}deg_run{run}.pdf', dpi=200)
            elif runtype=='bkg' and norm==True:
                plt.savefig(f'./plots/{campaign}normalized_{runtype}_DCRvTp0210_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}normalized_{runtype}_DCRvTp0210_run{run}.pdf', dpi=200)
            elif runtype=='alp' and norm==False:
                plt.savefig(f'./plots/{campaign}{runtype}_DCRvTp0210_{radius_fn}mm_{angle_det}deg_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}{runtype}_DCRvTp0210_{radius_fn}mm_{angle_det}deg_run{run}.pdf', dpi=200)
            elif runtype=='bkg' and norm==False:
                plt.savefig(f'./plots/{campaign}{runtype}_DCRvTp0210_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}{runtype}_DCRvTp0210_run{run}.pdf', dpi=200)
            # plt.show()
            plt.clf()
            plt.close()
            
        # DCR vs tp_0220___________
        if 'tp0220_v_DCR' in plot_list:

            fig, ax = plt.subplots()
            fig.suptitle(f'DCR vs 2-20% rise time', horizontalalignment='center', fontsize=16)

            tlo, thi, tpb = 0, 500, 10

            if run>=36 and run<117:
                dlo, dhi = -40, 170
                d_bins = 200
            elif run>=117:
                # dlo, dhi, dpb = -20., 40, 0.1
                dlo, dhi = -40, 170
                d_bins = 200

            nby = int((thi-tlo)/tpb)

            DCRvTp050_hist, xedges, yedges = np.histogram2d(df_cut['dcr_plot'], df_cut['tp0220'], bins=[d_bins, nby], range=([dlo, dhi], [tlo, thi]))
            X, Y = np.mgrid[dlo:dhi:d_bins*1j, tlo:thi:nby*1j]

            DCRvTp0210_hist_norm = np.divide(DCRvTp050_hist, (rt_min))

            pcm = plt.pcolormesh(X, Y, DCRvTp0210_hist_norm, shading='nearest', norm=LogNorm(0.001, 10))

            cb = plt.colorbar()
            if norm==True:
                cb.set_label("counts/min", ha = 'right', va='center', rotation=270, fontsize=14)
            else:
                cb.set_label("counts", ha = 'right', va='center', rotation=270, fontsize=14)
            cb.ax.tick_params(labelsize=12)
            ax.set_xlabel('DCR (arb)', fontsize=16)
            ax.set_ylabel('tp 02-20 (ns)', fontsize=16)
            plt.setp(ax.get_xticklabels(), fontsize=14)
            plt.setp(ax.get_yticklabels(), fontsize=14)

            # plt.legend()
            ax.text(0.95, 0.81, f'r = {radius} mm', verticalalignment='bottom',
                        horizontalalignment='right', transform=ax.transAxes, color='green', fontsize=14, bbox={'facecolor': 'white', 'alpha': 0.95, 'pad': 10})

            plt.title(f'{runtype} run {run}, {rt_min:.2f} mins', fontsize=12)
            plt.tight_layout()
            # plt.savefig(f'./plots/normScan/cal_normScan/{runtype}_dcr_vs_tp0_50_run{run}.png', dpi=200)
            if runtype=='alp' and norm==True:
                plt.savefig(f'./plots/{campaign}normalized_{runtype}_DCRvTp0220_{radius_fn}mm_{angle_det}deg_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}normalized_{runtype}_DCRvTp0220_{radius_fn}mm_{angle_det}deg_run{run}.pdf', dpi=200)
            elif runtype=='bkg' and norm==True:
                plt.savefig(f'./plots/{campaign}normalized_{runtype}_DCRvTp0220_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}normalized_{runtype}_DCRvTp0220_run{run}.pdf', dpi=200)
            elif runtype=='alp' and norm==False:
                plt.savefig(f'./plots/{campaign}{runtype}_DCRvTp0220_{radius_fn}mm_{angle_det}deg_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}{runtype}_DCRvTp0220_{radius_fn}mm_{angle_det}deg_run{run}.pdf', dpi=200)
            elif runtype=='bkg' and norm==False:
                plt.savefig(f'./plots/{campaign}{runtype}_DCRvTp0220_run{run}.png', dpi=200)
                plt.savefig(f'./plots/{campaign}{runtype}_DCRvTp0220_run{run}.pdf', dpi=200)
            # plt.show()
            plt.clf()
            plt.close()




def plot_energy(runs, etype='trapEftp', corr_DCR=True, corr_AoE=True, user=True, hit=True, cal=True):
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

    if cal==True:
            #etype_cal = etype+'_cal'
            etype+='_cal'

    for run in runs:


        df, runtype, rt_min, radius, angle_det, rotary = cage_utils.getDataFrame(run, user=user, hit=hit, cal=cal)

        # use baseline cut
        if run <79:
            bl_cut_lo, bl_cut_hi = 9150,9320
        if run>79 and run <117:
            bl_cut_lo, bl_cut_hi = 8500, 10000
        if run>=117:
            bl_cut_lo, bl_cut_hi = 9700, 9760

        df_cut = df.query(f'bl > {bl_cut_lo} and bl < {bl_cut_hi}').copy()

        # create new new DCR

        if corr_DCR==True and run>57:
            const, offset = cage_utils.corrDCR(df_cut, etype, e_bins=300, elo=0, ehi=6000, dcr_fit_lo=-30, dcr_fit_hi=40)
            df_cut['dcr_plot'] = df_cut['dcr']-offset + ((-1*const))*df_cut[etype]
        elif corr_DCR==True and run<57:
            const = const = 0.0011
            df_cut['dcr_plot'] = df_cut['dcr'] - const*df_cut[etype]
        else:
            df_cut['dcr_plot'] = df_cut['dcr']

        if corr_AoE==True:
            nb_AoE = 1000
            alo, ahi = 0.005, 0.075
            AoE_1d_hist, AoE_1d_bins, AoE_vars = pgh.get_hist(df_cut['AoE'], bins=nb_AoE, range=[alo, ahi])
            AoE_pars, AoE_cov = pgf.gauss_mode_width_max(AoE_1d_hist, AoE_1d_bins, AoE_vars)
            AoE_mode = AoE_pars[0]
            df_cut['AoE_plot'] = df_cut['AoE'] - AoE_mode

        else:
            df_cut['AoE_plot'] = df_cut['AoE']


        #create 0-50
        df_cut['tp0_50'] = df_cut['tp_50']- df_cut['tp_0']

        # create cut for alphas
        alpha_cut = f'dcr_plot > 25 and dcr_plot < 150 and tp0_50 > 150 and tp0_50 < 400 and {etype} >500 and {etype} < 5000'
        if run < 57:
            alpha_cut = f'dcr_plot > 35 and dcr_plot < 150 and tp0_50 > 150 and tp0_50 < 400 and {etype} >500 and {etype} < 4700'
        new_dcr_cut = df_cut.query(alpha_cut).copy()

        alpha_energy = np.array(new_dcr_cut[etype])
        mean_energy = np.mean(alpha_energy)
        std_energy = np.std(alpha_energy)
#         std_energy = np.sqrt(len(new_dcr_cut['trapEmax']))

        alpha_dcr = np.array(new_dcr_cut['dcr_plot'])
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

    plt.savefig('./plots/new_normScan/errorbars_energy_deg.png', dpi=200)

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

    plt.savefig('./plots/new_normScan/errorbars_dcr_avg.png', dpi=200)

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

    plt.savefig('./plots/new_normScan/energy_deg.png', dpi=200)

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

    plt.savefig('./plots/new_normScan/dcr_avg.png', dpi=200)

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
