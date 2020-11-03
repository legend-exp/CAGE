#!/usr/bin/env python3
# Written by Gulden Othman September 2020.
# Calculates capacitance from data taking while biasing with a pulser
import numpy as np
import sys
import math
import matplotlib.pyplot as plt
import pandas as pd

def main():

    data, title, runtime = linear_edge()
    # data = linear_edge()

    # plot_rotary(data, title, runtime)
    plot_linear(data, title, runtime)

def rotary():
    # elog 220
    title = 'Rotary Scan at 22 mm radius'
    runtime = 30. # 30 min runs
    data = {'linear': [22, 22, 22, 22, 22, 22, 22, 22],
            'rotary': [0, -15, -30, -45, -60, -75, -90, -135],
            'source': [-180, -180, -180, -180, -180, -180, -180, -180],
            'peak_cts': [475, 459, 470, 428, 507, 479, 618, 838],
            'peak_left': [148, 148, 148, 148, 148, 148, 147, 147],
            'peak_right': [152, 152, 152, 152, 152, 152, 152, 152],
            'bkg_cts': [626, 612, 597, 603, 619, 613, 659, 638],
            'bkg_left': [139, 139, 139, 139, 139, 139, 138, 138],
            'bkg_right': [148, 148, 148, 148, 148, 148, 147, 147]}

    df = pd.DataFrame(data, columns = ['linear', 'rotary', 'source', 'peak_cts',
                                       'peak_left', 'peak_right', 'bkg_cts',
                                       'bkg_left', 'bkg_right'])
    return df, title, runtime

def rotary_edge():
    # elog 230
    title = 'Rotary Edge Scan at -135 deg'
    runtime = 30.# 30 min runs
    data_135 = {'linear': [22, 23, 24, 25],
            'rotary': [-135, -135, -135, -135],
            'source': [-180, -180, -180, -180],
            'peak_cts': [858, 886, 840, 727],
            'peak_left': [147, 147, 147, 146],
            'peak_right': [154, 154, 154, 153],
            'bkg_cts': [1409, 1361, 1329, 1304],
            'bkg_left': [126, 126, 126, 125],
            'bkg_right': [147, 147, 147, 146]}

    data_90 = {'linear': [22, 23, 24, 25],
            'rotary': [-90, -90, -90, -90],
            'source': [-180, -180, -180, -180],
            'peak_cts': [771, 732, 677, 738],
            'peak_left': [146, 146, 146, 146],
            'peak_right': [153, 153, 153, 153],
            'bkg_cts': [1436, 1431, 1394, 1312],
            'bkg_left': [125, 125, 125, 125],
            'bkg_right': [146, 146, 146, 146]}

    df = pd.DataFrame(data_135, columns = ['linear', 'rotary', 'source', 'peak_cts',
                                       'peak_left', 'peak_right', 'bkg_cts',
                                       'bkg_left', 'bkg_right'])
    return df, title, runtime


def linear_coarse():
    # elog 212
    title = 'Coarse Linear Scan at 0 deg rotary'
    runtime = 15. # 15 min runs
    data = {'linear': [0, 5, 10, 15, 20, 25],
            'rotary': [0, 0, 0, 0, 0, 0],
            'source': [-180, -180, -180, -180, -180, -180],
            'peak_cts': [458, 485, 437, 537, 588, 402],
            'peak_left': [150, 150, 150, 149, 148, 148],
            'peak_right': [158, 158, 158, 157, 157, 157],
            'bkg_cts': [786, 853, 793, 844, 965, 884],
            'bkg_left': [125, 124, 124, 124, 121, 121],
            'bkg_right': [149, 148, 148, 148, 148, 148]}
    df = pd.DataFrame(data, columns = ['linear', 'rotary', 'source', 'peak_cts',
                                       'peak_left', 'peak_right', 'bkg_cts',
                                       'bkg_left', 'bkg_right'])
    return df, title, runtime

def linear_edge():
    # elog 215
    title = 'Linear Edge Scan at 0 deg rotary'
    runtime = 15. # 15 minute runs
    data = {'linear': [20, 21, 22, 23, 24, 25],
            'rotary': [0, 0, 0, 0, 0, 0],
            'source': [-180, -180, -180, -180, -180, -180],
            'peak_cts': [570, 457, 429, 361, 344, 337],
            'peak_left': [153, 152, 152, 152, 152, 152],
            'peak_right': [162, 161, 161, 161, 161, 161],
            'bkg_cts': [830, 851, 802, 713, 721, 724],
            'bkg_left': [126, 125, 125, 128, 128, 128],
            'bkg_right': [153, 152, 152, 152, 152, 152]}

    df = pd.DataFrame(data, columns = ['linear', 'rotary', 'source', 'peak_cts',
                                       'peak_left', 'peak_right', 'bkg_cts',
                                       'bkg_left', 'bkg_right'])
    return df, title, runtime

def plot_rotary(data, title, runtime):
    df = data
    print(data)

    linear = np.array(df['linear'])
    rotary = np.array(df['rotary'])

    peak_cts = np.array(df['peak_cts'])
    peak_left = np.array(df['peak_left'])
    peak_right = np.array(df['peak_right'])

    bkg_cts = np.array(df['bkg_cts'])
    bkg_left = np.array(df['bkg_left'])
    bkg_right = np.array(df['bkg_right'])

    final_cts = np.zeros(len(linear))
    uncertainty = np.zeros(len(linear))

    for i in range(len(linear)):
        peak_bins = peak_right[i] - peak_left[i]
        bkg_bins = bkg_right[i] - bkg_left[i]
        bin_ratio = bkg_bins/peak_bins
        final_cts[i] = peak_cts[i] - bkg_cts[i]/bin_ratio
        uncertainty[i] = np.sqrt(peak_cts[i]) + np.sqrt(bkg_cts[i])/bin_ratio

    norm_cts = final_cts/runtime
    uncertainty_norm = uncertainty/runtime

    fig = plt.figure()
    plt.errorbar(rotary, norm_cts, yerr=uncertainty_norm, marker = '.', ls='none')
    plt.title(title)
    plt.xlabel('Rotary position (deg)')
    plt.ylabel('Background Subtracted Counts/min')
    plt.show()

def plot_linear(data, title, runtime):
    df = data
    print(data)

    linear = np.array(df['linear'])

    peak_cts = np.array(df['peak_cts'])
    peak_left = np.array(df['peak_left'])
    peak_right = np.array(df['peak_right'])

    bkg_cts = np.array(df['bkg_cts'])
    bkg_left = np.array(df['bkg_left'])
    bkg_right = np.array(df['bkg_right'])

    final_cts = np.zeros(len(linear))
    uncertainty = np.zeros(len(linear))

    for i in range(len(linear)):
        peak_bins = peak_right[i] - peak_left[i]
        bkg_bins = bkg_right[i] - bkg_left[i]
        bin_ratio = bkg_bins/peak_bins
        final_cts[i] = peak_cts[i] - bkg_cts[i]/bin_ratio
        uncertainty[i] = np.sqrt(peak_cts[i]) + np.sqrt(bkg_cts[i])/bin_ratio

    norm_cts = final_cts/runtime
    uncertainty_norm = uncertainty/runtime
    # print(norm_cts)
    # exit()

    fig = plt.figure()
    plt.errorbar(linear, norm_cts, yerr=uncertainty_norm, marker = '.', ls='none')
    plt.title(title)
    plt.xlabel('linear position (mm)')
    plt.ylabel('Background Subtracted Counts/min')
    plt.show()


if __name__ == '__main__':
	main()
