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
    base_filename = '../alpha/processed_out/oppi/processed'
    processed_filename = '../alpha/processed_out/oppi/processed_oppi_ring_y10_norm_241Am_100000000.hdf5'

    primaries = 10000000
    radius = [5, 6, 7, 8, 9, 10] # in mm
    rotary_angles = [0, -145, -180]
    elo = 0.05 # in MeV
    ehi = 0.07 # in MeV

    # getCounts(processed_filename) # get all counts in physical volume for this file. Useful for debugging if sim was successful
    # getCounts_cut(processed_filename, elo, ehi) # get counts within specific energy region
    # getRate(processed_filename, primaries, elo, ehi) # get rate in counts/sec for specific energy region
    plotRate(radius, rotary_angles, elo, ehi, rotary=True) # plot rates for multiple source positions (sims files) on one plot

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

if __name__ == '__main__':
	main()
