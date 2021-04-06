import numpy as np
import scipy
import matplotlib as mpl
from matplotlib.colors import LogNorm
from scipy.stats import norm, kde
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import sys
from particle import PDGID
mpl.rcParams['text.usetex'] = True
mpl.use('Agg')

def main():
    base_filename = '../alpha/processed_out/oppi/'
    processed_filename = '../alpha/processed_out/oppi/rotary_centering_scan/processed_y12_thetaDet90_rotary4_241Am_100000000.hdf5'
    run = 'angled_rotary_centering_scan/'
    processed_dir = f'{base_filename}{run}'

    primaries = 10000000
    radius = [12] # in mm
    rotary_angles = np.linspace(4, 144, 15)
#     elo = 5.4 # in MeV
#     ehi = 5.6 # in MeV
    elo = 0. # in MeV
    ehi = 6. # in MeV

    # getCounts(processed_filename) # get all counts in physical volume for this file. Useful for debugging if sim was successful
    # getCounts_cut(processed_filename, elo, ehi) # get counts within specific energy region
    getRate(processed_filename, primaries, elo, ehi) # get rate in counts/sec for specific energy region
    # plotRate(radius, rotary_angles, elo, ehi, rotary=True) # plot rates for multiple source positions (sims files) on one plot
    # rotary_plotRate(radius, rotary_angles, elo, ehi, processed_dir) # for rotary scans, plot rates for multiple source positions (sims files) on one plot

def getCounts(processed_filename):
    df = pd.read_hdf(processed_filename, keys='procdf')
    energy = np.array(df['energy'])
    counts = len(energy)
    print('%f counts in PV' %counts)

def getCounts_cut(processed_filename, elo, ehi):
    df = pd.read_hdf(processed_filename, keys='procdf')
    energy = np.array(df['energy'])
    cut_df = df.loc[(df.energy > elo) & (df.energy < ehi)]
    cut_energy_keV = np.array(cut_df['energy']*1000)
    counts = len(cut_energy_keV)
    print(f'{counts} counts in region {elo} to {ehi} keV')

    return(counts)

def getRate(processed_filename, primaries, elo, ehi):
    # see this elog https://elog.legend-exp.org/UWScanner/166
    source_activity = 4.0e4 #40 kBq = 4e4 decays/s
    time_seconds = primaries/(source_activity)
    counts = getCounts_cut(processed_filename, elo, ehi)
    rate = counts/time_seconds #(rate in counts/s)
    rate_err = np.sqrt(counts)/time_seconds
    print(f'{rate} counts/second in region {elo} to {ehi} keV')

    return(rate, rate_err)

def plotRate(radius, rotary_angles, elo, ehi, rotary=False):
    rates_arr = []
    rates_uncertainty = []
    fig, ax = plt.subplots(figsize=(6,5))

    if rotary==True:
#         cmap = plt.cm.get_cmap('jet', len(rotary_angles))
        cmap = ['g', 'b', 'r']
        for rot, i in zip (rotary_angles, range(len(rotary_angles))):
            rates_arr = []
            rates_uncertainty = []
            for r in radius:
                rate, rate_err = getRate(f'../alpha/processed_out/oppi/centering_scan/processed_y{r}_norm_rotary{rot}_241Am_100000000.hdf5', 10000000, elo, ehi)
                rates_arr.append(rate)
                rates_uncertainty.append(rate_err)
                
            plt.errorbar(radius, rates_arr, yerr=rates_uncertainty, marker = '.', c=cmap[i], ls='none', label=f'rotary: {rot} deg')
            
#     print(rates_arr)

#     fig, ax = plt.subplots(figsize=(6,5))
#     plt.errorbar(radius, rates_arr, yerr=rates_uncertainty, marker = '.', ls='none')
#     plt.plot(radius, rates_arr, '.r')
    plt.xlabel('Radius (mm)', fontsize=16)
    plt.ylabel('Rate (cts/sec)', fontsize=16)
    plt.title(f'Rate for {elo} to {ehi} MeV', fontsize=16)
    plt.setp(ax.get_xticklabels(), fontsize=14)
    plt.setp(ax.get_yticklabels(), fontsize=14)
    plt.legend()
    plt.savefig(f'./rates_rotary_{elo}_{ehi}.png',  dpi=200)
    return(rate)

def rotary_plotRate(radius, rotary_angles, elo, ehi, processed_dir):
    rates_arr = []
    rates_uncertainty = []
    fig, ax = plt.subplots(figsize=(6,5))

    cmap = plt.cm.get_cmap('jet', len(radius))
    cmap = ['b', 'g', 'r']
    for r, i in zip (radius, range(len(radius))):
        rates_arr = []
        rates_uncertainty = []
        for rot in rotary_angles:
            rate, rate_err = getRate(f'{processed_dir}processed_y{r}_thetaDet61_rotary{int(rot)}_241Am_100000000.hdf5', 10000000, elo, ehi)
            rates_arr.append(rate)
            rates_uncertainty.append(rate_err)
                
    plt.errorbar(rotary_angles, rates_arr, yerr=rates_uncertainty, marker = '.', c=cmap[i], ls='none', label=f'radius: {r} mm')
            
#     print(rates_arr)

#     fig, ax = plt.subplots(figsize=(6,5))
#     plt.errorbar(radius, rates_arr, yerr=rates_uncertainty, marker = '.', ls='none')
#     plt.plot(radius, rates_arr, '.r')
    plt.xlabel('Rotary Position (deg)', fontsize=16)
    plt.ylabel('Rate (cts/sec)', fontsize=16)
    plt.title(f'Rate for {elo} to {ehi} MeV', fontsize=16)
    plt.setp(ax.get_xticklabels(), fontsize=14)
    plt.setp(ax.get_yticklabels(), fontsize=14)
    plt.legend()
    plt.savefig(f'./rates_angled_rotary_centering_{elo}_{ehi}.png',  dpi=200)
    return(rate)

if __name__ == '__main__':
	main()
