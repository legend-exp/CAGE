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
plt.style.use('./joule_dissertation.mplstyle')

mpl.use('Agg')

def main():  
    user = True
    hit = True
    cal = True
    
    run=64
    
    
    cut1 = 'trapEftp_cal > 1458 and trapEftp_cal < 1462'
    cut2 = 'trapEftp_cal > 59.0 and trapEftp_cal < 60.0'
    
    # for alpha example
    # cut1 = 'trapEftp_cal > 3900 and trapEftp_cal < 4100 and dcr > 90 and dcr < 100'
    
    
    plot_wfs(run, cut2, user=user, hit=hit, cal=cal, norm=True)
    # compare_wfs(run, cut1, cut2, user=user, hit=hit, cal=cal, norm=True)
    # wfs_filtered(run, cut1, user=user,1hit=hit, cal=cal, norm=False)

    
def plot_wfs(run, cut, user=True, hit=True, cal=True, norm=False):
    df_raw, dg, runtype, rt_min, radius, angle_det, rotary = cage_utils.getDataFrame(run, user=user, hit=hit,
                                                                                         cal=cal)
    df = cage_utils.apply_DC_Cuts(run, df_raw)
    
    
    
    #times, super_wf = cage_utils.get_superpulse_taligned(df, dg, cut, nwfs=10, norm=False)
    times, wfs = cage_utils.get_superpulse_taligned(df, dg, cut, nwfs=20, norm=norm)
    
    print(len(wfs))
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax = plt.axes()
    
    ax.plot(times, wfs)
        

    plt.title(f'1460 keV waveform')

    plt.xlabel('clock cycles')
    plt.ylabel('normalized ADU')
    
    plt.savefig('./plots/1460_wf.png', dpi=200)
    plt.savefig('./plots/1460_wf.pdf', dpi=200)
    
    # plt.xlim(3800, 8000)
    # plt.ylim(3000, 4200)
    
    # plt.savefig('./plots/60_wfs_tail.png', dpi=200)
    # plt.savefig('./plots/60_wfs_tail.pdf', dpi=200)
    
    plt.xlim(3500, 4100)
    plt.savefig('./plots/1460_wf_rise.png', dpi=200)
    plt.savefig('./plots/1460_wf_rise.pdf', dpi=200)
    

def compare_wfs(run, cut1, cut2, user=True, hit=True, cal=True, norm=False):
    df_raw, dg, runtype, rt_min, radius, angle_det, rotary = cage_utils.getDataFrame(run, user=user, hit=hit,
                                                                                         cal=cal)
    df = cage_utils.apply_DC_Cuts(run, df_raw)
    
    
    
    #times, super_wf = cage_utils.get_superpulse_taligned(df, dg, cut, nwfs=10, norm=False)
    times, wfs1 = cage_utils.get_wfs(df, dg, cut1, nwfs=1, norm=norm)
    times, wfs2 = cage_utils.get_wfs(df, dg, cut2, nwfs=1, norm=norm)
    
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax = plt.axes()
    
    
    ax.plot(times, wfs2[0], c='r', label='60 keV')
    ax.plot(times, wfs1[0], c='g', label='1460 keV')
        

    plt.title(f'Comparison of 1460 keV to 60 keV waveforms')

    plt.xlabel('clock cycles')
    plt.ylabel('normalized ADU')
    
    plt.legend()
    
    plt.savefig('./plots/1460_v_60keV_wfs.png', dpi=200)
    plt.savefig('./plots/1460_v_60keV_wfs.pdf', dpi=200)
    
#     plt.xlim(3800, 8000)
#     plt.ylim(3000, 4200)
    
#     plt.savefig('./plots/1460_v_60keV_wfs_tail.png', dpi=200)
#     plt.savefig('./plots/1460_v_60keV_wfs_tail.pdf', dpi=200)

def wfs_filtered(run, cut1, user=True, hit=True, cal=True, norm=False):
    df_raw, dg, runtype, rt_min, radius, angle_det, rotary = cage_utils.getDataFrame(run, user=user, hit=hit,
                                                                                         cal=cal)
    df = cage_utils.apply_DC_Cuts(run, df_raw)
    
    
    # times, wf = cage_utils.get_wfs(df, dg, cut1, nwfs=1, norm=norm)
    times, wf = cage_utils.get_wfs(df, dg, cut1, nwfs=1, norm=norm, tp_align=0.5, n_pre = 3800, n_post = 4175)
    

    wf_dcr_pz = cage_utils.double_pole_zero(wf, 21250, 433, 0.045)[0]
    wf_pz = cage_utils.double_pole_zero(wf, 18750, 317, 0.035)[0] 
    Etrap = cage_utils.trap_norm(wf_pz, 100, 400)
    dcr_trap = cage_utils.trap_norm(wf_dcr_pz, 750, 2250)
    wf_atrap = cage_utils.asymTrapFilter(wf_pz, 20, 100, 400)
    atrap_max = np.argmax(wf_atrap)
    t0 = cage_utils.time_point_thresh_max(wf_atrap, 10., atrap_max, 3500)
    Eftp = Etrap[t0 + 400]
    dcr = dcr_trap[7900]
    
    print(Eftp)
    print(t0)

    
    fig, ax = plt.subplots()
    # fig, ax = plt.subplots(figsize=(14, 10))
    ax = plt.axes()
    
    
    ax.plot(times, wf[0], c='b', label='wf_blsub')
    ax.plot(times, wf_pz, c='r', label='wf_pz')
    ax.plot(times, Etrap, c='g', label='wf_trap')
    plt.axhline(Eftp, lw=2, c = 'c', linestyle='dotted', label='trapEftp')
    plt.plot(t0+400, Eftp, '*y', markersize=20)
        

    plt.title(f'Energy determination')

    plt.xlabel('clock cycles')
    plt.ylabel('ADU')
    
    plt.legend(fontsize=16)
    
    plt.savefig('./plots/wf_etrap.png', dpi=200)
    plt.savefig('./plots/wf_etrap.pdf', dpi=200)
    
    plt.clf()
    plt.close()
    
    fig, ax = plt.subplots()
    ax = plt.axes()
    
    fig, ax = plt.subplots()
    # fig, ax = plt.subplots(figsize=(14, 10))
    ax = plt.axes()
    
    
    ax.plot(times, wf[0], c='b', label='wf_blsub')
    ax.plot(times, wf_atrap, c='g', label='wf_atrap')
    plt.axvline(t0, lw=2, c = 'r', label='tp_0')
        

    plt.title(f'tp_0 determination')

    plt.xlabel('clock cycles')
    plt.ylabel('ADU')
    
    plt.legend()
    
    plt.savefig('./plots/wf_t0.png', dpi=200)
    plt.savefig('./plots/wf_t0.pdf', dpi=200)
    
    plt.xlim(3720, 3840)
    
    plt.savefig('./plots/wf_t0_zoom.png', dpi=200)
    plt.savefig('./plots/wf_t0_zoom.pdf', dpi=200)
    
    plt.clf()
    plt.close()
    
    fig, ax = plt.subplots()
    ax = plt.axes()
    
    ax.plot(times, wf[0], c='b', label='wf_blsub')
    ax.plot(times, wf_dcr_pz, c='r', label='wf_pzDCR')
    ax.plot(times, dcr_trap, c='g', label='dcr_trap')
    plt.axhline(dcr, lw=2, c = 'c', linestyle='dotted', label='dcr')
    plt.plot(7900, dcr, '*y', markersize=20)
    #ax.axvspan(7899, 7901, alpha=0.1, color='g', label='dcr_')
        

    plt.title(f'DCR determination')
    plt.xlabel('clock cycles')
    plt.ylabel('ADU')
    
    plt.legend()
    plt.tight_layout()
    
    plt.savefig('./plots/wf_dcr.png', dpi=200)
    plt.savefig('./plots/wf_dcr.pdf', dpi=200)
    
    # plt.ylim(-1, 50) # for 1460 keV
    plt.ylim(-1, 120) # for alpha
    
    plt.savefig('./plots/wf_dcr_bl.png', dpi=200)
    plt.savefig('./plots/wf_dcr_bl.pdf', dpi=200)
    
    plt.clf()
    plt.close()


    

if __name__=="__main__":
    main()