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
    radius = [5, 6, 7, 8, 10] # in mm
    rotary_angles = [-145, -180]
    elo = 0.05 # in MeV
    ehi = 0.07 # in MeV

    # getCounts(processed_filename) # get all counts in physical volume for this file. Useful for debugging if sim was successful
    # getCounts_cut(processed_filename, elo, ehi) # get counts within specific energy region
    # getRate(processed_filename, primaries, elo, ehi) # get rate in counts/sec for specific energy region
    plotRate(radius, elo, ehi) # plot rates for multiple source positions (sims files) on one plot

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

def plotRate(radius, rotary_angles, rotary=False, elo, ehi):
    rates_arr = []
    rates_uncertainty = []
    radius_arr = []
    rotary_array = []
    names = ['radius', 'rotary', 'rate', 'rate_err']
    # df = pd.df(columns=names)
    # for r in radius:
    #     rate, rate_err = getRate(f'../alpha/processed_out/oppi/processed_oppi_largeHole_ring_y{r}_norm_241Am_100000000.hdf5', 10000000, elo, ehi)
    #     rates_arr.append(rate)
    #     rates_uncertainty.append(rate_err)


    if rotary==True:
        for r, rot in zip(radius, rotary_angles):
            print(r, rot)
            # rate, rate_err = getRate(f'../alpha/processed_out/oppi/processed_y{r}_norm_rotary{rot}_241Am_100000000.hdf5', 10000000, elo, ehi)
            # rates_arr.append(rate)
            # rates_uncertainty.append(rate_err)
            # radius_array.append(r)
            # rotary_array.append(rot)

#
#     print(rates_arr)
#
#     fig, ax = plt.subplots(figsize=(6,5))
#     plt.errorbar(radius, rates_arr, yerr=rates_uncertainty, marker = '.', ls='none')
# #     plt.plot(radius, rates_arr, '.r')
#     plt.xlabel('Radius (mm)')
#     plt.ylabel('Rate (cts/sec)')
#     plt.title(f'Rate for {elo} to {ehi} MeV \n larger than nominal hole')
#     plt.savefig(f'./rates_largeHole_{elo}_{ehi}.png')
    #return(rate)

if __name__ == '__main__':
	main()
