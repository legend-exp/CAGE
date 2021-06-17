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

import pygama
from pygama import DataGroup
import pygama.lh5 as lh5
import pygama.analysis.histograms as pgh
import pygama.analysis.peak_fitting as pgf


def getDataFrame(run, user=True, hit=True, cal=True, lowE=False, dsp_list=[]):
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
        print(f'Radius: {radius}; Angle: {angle_det}; Rotary: {rotary}')

    else:
        radius = 'n/a'
        angle = 'n/a'
        angle_det = 'n/a'
        rotary = 'n/a'


    # print(etype, etype_cal, run)
    # exit()

    print(f'user: {user}; cal: {cal}; hit: {hit}')



    # get data and load into df
    lh5_dir = dg.lh5_user_dir if user else dg.lh5_dir
    
    if cal==True:
        default_dsp_list = ['energy', 'trapEmax', 'trapEftp', 'trapEftp_cal', 'bl','bl_sig', 'bl_slope', 'lf_max', 'A_10','AoE', 'dcr', 'tp_0', 'tp_10', 'tp_90', 'tp_50', 'tp_80', 'tp_max', 'ToE']
        
    else:
        default_dsp_list = ['energy', 'trapEmax', 'trapEftp', 'bl','bl_sig', 'bl_slope', 'lf_max', 'A_10','AoE', 'dcr', 'tp_0', 'tp_10', 'tp_90', 'tp_50', 'tp_80', 'tp_max', 'ToE']
    
    if len(dsp_list) < 1:
        print(f'No options specified for DSP list! Using default: {default_dsp_list}')
        dsp_list = default_dsp_list
    
    

    if hit==True:
        print('Using hit files')
        file_list = lh5_dir + dg.fileDB['hit_path'] + '/' + dg.fileDB['hit_file']
        
        if lowE==True:
            file_list = lh5_dir + dg.fileDB['hit_path'] + '/lowE/' + dg.fileDB['hit_file']
            print(f'Using lowE calibration files \n {file_list}')
            
        if cal==True:
            df = lh5.load_dfs(file_list, dsp_list, 'ORSIS3302DecoderForEnergy/hit')


        if cal==False:
            df = lh5.load_dfs(file_list, dsp_list, 'ORSIS3302DecoderForEnergy/hit')


    elif hit==False:
        print('Using dsp files')
        file_list = lh5_dir + dg.fileDB['dsp_path'] + '/' + dg.fileDB['dsp_file']
        if cal==True:
            df = lh5.load_dfs(file_list, dsp_list, 'ORSIS3302DecoderForEnergy/dsp')
        if cal==False:
            df = lh5.load_dfs(file_list, dsp_list, 'ORSIS3302DecoderForEnergy/dsp')

    else:
        print('dont know what to do here! need to specify if working with calibrated/uncalibrated data, or dsp/hit files')



    return(df, runtype, rt_min, radius, angle_det, rotary)

def getStartStop(run):
    """
    get the start time and stop time for a given run
    """
    
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
    
    u_stop = u_start + rt_min*60
    
    t_stop = pd.to_datetime(u_stop, unit='s')
    
    print(f'start: {t_start}\n stop: {t_stop}')
    return(t_start, t_stop)

def corrDCR(df, etype, e_bins=300, elo=0, ehi=6000, dcr_fit_lo=-30, dcr_fit_hi=30):

    df_dcr_cut = df.query(f'dcr >{dcr_fit_lo} and dcr < {dcr_fit_hi} and {etype} > {elo} and {etype} < {ehi}').copy()


    median, xedges, binnumber = stats.binned_statistic(df_dcr_cut[etype], df_dcr_cut['dcr'], statistic = "median", bins = e_bins)

    en_bin_centers = pgh.get_bin_centers(xedges)

    fit_raw, cov = np.polyfit(en_bin_centers, median, deg=1, cov=True)

    const = fit_raw[0]
    offset = fit_raw[1]
    err = np.sqrt(np.diag(cov))

    print(f'Fit results\n slope: {const}\n offset: {offset}')
    return(const, offset, err)

def mode_hist(df, param, a_bins=1000, alo=0.005, ahi=0.075, cut=False, cut_str=''):
    # get the mode of a section of a histogram. Default params based on AoE values
    if cut==True:
        print(f'Using cut before finding mode: {cut_str}')
        df_plot = df.query(cut_str)
    else:
        df_plot = df
    hist, bins, vars = pgh.get_hist(df_plot[param], bins=a_bins, range=[alo, ahi])
    pars, cov = pgf.gauss_mode_width_max(hist, bins, vars)
    mode = pars[0]

    return(mode)


def testFunc(list):
    if 'test' in list:
        print("hi it worked")
