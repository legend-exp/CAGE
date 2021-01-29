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

from pygama import DataGroup
import pygama.lh5 as lh5
import pygama.analysis.histograms as pgh
import pygama.analysis.peak_fitting as pgf

mpl.use('Agg')

def main():
    run = 136
    cycle = 1481
    user = False
    hit = False
    cal = False
    etype = 'trapEftp'

    plot_wfs(run, cycle, user, hit, cal)

def plot_wfs(run, cycle, etype, user=False, hit=True, cal=True):
    """
    show waveforms in different enery regions.
    use the dsp or hit file to select events
    """
    dg = DataGroup('$CAGE_SW/processing/cage.json', load=True)
    str_query = f'cycle=={cycle} and skip==False'
    dg.fileDB.query(str_query, inplace=True)

    #get runtime, startime, runtype
    runtype_list = np.array(dg.fileDB['runtype'])
    runtype = runtype_list[0]
    rt_min = dg.fileDB['runtime'].sum()
    u_start = dg.fileDB.iloc[0]['startTime']
    t_start = pd.to_datetime(u_start, unit='s')


    # get data and load into df
    lh5_dir = dg.lh5_user_dir if user else dg.lh5_dir
    if cal==True:
        etype_cal = etype+'_cal'

    if hit==True:
        print('Using hit files')
        file_list = lh5_dir + dg.fileDB['hit_path'] + '/' + dg.fileDB['hit_file']
        if run<=117 and cal==True:
            df = lh5.load_dfs(file_list, [f'{etype}', f'{etype_cal}', 'bl','bl_sig','A_10','AoE', 'ts_sec', 'dcr_raw', 'dcr_ftp', 'dcr_max', 'tp_0', 'tp_10', 'tp_90', 'tp_50', 'tp_80', 'tp_max'], 'ORSIS3302DecoderForEnergy/hit')
        elif run>117 and cal==True:
            df = lh5.load_dfs(file_list, [f'{etype}', f'{etype_cal}', 'bl','bl_sig', 'bl_slope', 'lf_max', 'A_10','AoE', 'dcr', 'tp_0', 'tp_10', 'tp_90', 'tp_50', 'tp_80', 'tp_max'], 'ORSIS3302DecoderForEnergy/hit')

        elif run<=117 and cal==False:
            df = lh5.load_dfs(file_list, [f'{etype}', 'bl','bl_sig','A_10','AoE', 'ts_sec', 'dcr_raw', 'dcr_ftp', 'dcr_max', 'tp_0', 'tp_10', 'tp_90', 'tp_50', 'tp_80', 'tp_max'], 'ORSIS3302DecoderForEnergy/hit')
        elif run>117 and cal==False:
            df = lh5.load_dfs(file_list, [f'{etype}', 'bl','bl_sig', 'bl_slope', 'lf_max', 'A_10','AoE', 'dcr', 'tp_0', 'tp_10', 'tp_90', 'tp_50', 'tp_80', 'tp_max'], 'ORSIS3302DecoderForEnergy/hit')

    elif hit==False:
        print('Using dsp files')
        file_list = lh5_dir + dg.fileDB['dsp_path'] + '/' + dg.fileDB['dsp_file'])
        if run<=117 and cal==True:
            df = lh5.load_dfs(file_list, [f'{etype}', f'{etype_cal}', 'bl','bl_sig','A_10','AoE', 'ts_sec', 'dcr_raw', 'dcr_ftp', 'dcr_max', 'tp_0', 'tp_10', 'tp_90', 'tp_50', 'tp_80', 'tp_max'], 'ORSIS3302DecoderForEnergy/dsp')
        elif run>117 and cal==True:
            df = lh5.load_dfs(file_list, [f'{etype}', f'{etype_cal}', 'bl','bl_sig', 'bl_slope', 'lf_max', 'A_10','AoE', 'dcr', 'tp_0', 'tp_10', 'tp_90', 'tp_50', 'tp_80', 'tp_max'], 'ORSIS3302DecoderForEnergy/dsp')

        elif run<=117 and cal==False:
            df = lh5.load_dfs(file_list, [f'{etype}', 'bl','bl_sig','A_10','AoE', 'ts_sec', 'dcr_raw', 'dcr_ftp', 'dcr_max', 'tp_0', 'tp_10', 'tp_90', 'tp_50', 'tp_80', 'tp_max'], 'ORSIS3302DecoderForEnergy/dsp')
        elif run>117 and cal==False:
            df = lh5.load_dfs(file_list, [f'{etype}', 'bl','bl_sig', 'bl_slope', 'lf_max', 'A_10','AoE', 'dcr', 'tp_0', 'tp_10', 'tp_90', 'tp_50', 'tp_80', 'tp_max'], 'ORSIS3302DecoderForEnergy/dsp')

    else:
        print('dont know what to do here! need to specify if working with calibrated/uncalibrated data, or dsp/hit files')


    waveforms = []

    n_eranges = 50
    nwfs= 50
    emin = 5000
    emax = 15000

    eranges = np.linspace(emin, emax, n_eranges) #trying to get about 60 keV energy slices
    for e in eranges:
        #get events within 1% of energy
        elo = e-(0.01*e)
        ehi = e+(0.01*e)
        idx = df[etype].loc[(df[etype] >= elo) & (df[etype] <= ehi)].index[:nwfs]
        raw_store = lh5.Store()
        tb_name = 'ORSIS3302DecoderForEnergy/raw'
        lh5_dir = dg.lh5_dir
        raw_list = lh5_dir + dg.fileDB['raw_path'] + '/' + dg.fileDB['raw_file']
        f_raw = raw_list.values[0] # fixme, only works for one file rn
        data_raw, nrows = raw_store.read_object(tb_name, f_raw, start_row=0, n_rows=idx[-1]+1)

        wfs_all = (data_raw['waveform']['values']).nda
        wfs = wfs_all[idx.values, :]
        # baseline subtraction
        bl_means = wfs[:,:800].mean(axis=1)
        wf_blsub = (wfs.transpose() - bl_means).transpose()
        ts = np.arange(0, wf_blsub.shape[1]-1, 1)
        super_wf = np.mean(wf_blsub, axis=0)
        wf_max = np.amax(super_wf)
        superpulse = np.divide(super_wf, wf_max)
        waveforms.append(superpulse)

    fig, ax = plt.subplots()
    ax = plt.axes()

    # set up colorbar to plot waveforms of different energies different colors
    colors = plt.cm.viridis(np.linspace(0, 1, n_eranges))
    c = np.arange(0, n_eranges)
    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
    cmap.set_array([])

    for n in range(n_eranges):
        plt.plot(ts, waveforms[n][:len(waveforms[n])-1], c=cmap.to_rgba(n))

    cb = fig.colorbar(cmap, ticks=list(eranges))
    cb.set_label("Energy", ha = 'right', va='center', rotation=270, fontsize=20)
    cb.ax.tick_params(labelsize=18)

    plt.xlim(3800, 8000)
    plt.ylim(0.4, 1.01)
    plt.setp(ax.get_xticklabels(), fontsize=18)
    plt.setp(ax.get_yticklabels(), fontsize=18)
    plt.title(f'Waveforms, {emin}-{emax} trapEftp, {n_eranges} steps', fontsize=20)
    plt.xlabel('clock cycles', fontsize=20)
    plt.savefig(f'./plots/angleScan/wfs_fallingEdge_run{run}.png', dpi=300)
    # plt.cla()

if __name__=="__main__":
    main()
