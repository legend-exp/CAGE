import os
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
    radius = [10]
    thetaDet = [90]
    rotAngle = [0]
    scan = 'large_hole_rate'
#   radius = [5, 6, 7, 8, 9, 10]
#   thetaDet = [90]
#   rotAngle = [162]
#   scan = 'spot_size_scan'
    processed_dir = f'../data/oppi/{scan}'
    primaries = 1e8
    source_activity = 4e4 #40 kBq = 4e4 decays/s
    activity_err = 4e4*0.3 # +- 30%
    time_seconds = primaries/(source_activity)

    r12_alp = []
    r14_alp = []
    angled_alp = []
    
    spot = []
    for r in radius:
        for theta in thetaDet:
            if r == 12 and theta == 71:
                continue
            for rot in rotAngle:
                name = f'y{r}_thetaDet{theta}_rotary{rot}'
                sixty = 0
                alphas = 0
                total = 0
                for file in os.listdir(f'{processed_dir}/{name}'):
                    file_name = f'{processed_dir}/{name}/{file}'
                    sixty += getCounts_cut(file_name, 0.058, 0.061)
                    alphas += getCounts_cut(file_name, 5.3, 5.5)
                    total += getCounts(file_name)
                    
                rate = total/time_seconds #(rate in counts/s)
                rate_sixty = sixty/time_seconds #(rate in counts/s)
                rate_alpha = alphas/time_seconds #(rate in counts/s)
                alp_err = (np.sqrt(alphas)*source_activity + alphas*activity_err)/primaries
                print(name + ':')
                print(f'{rate} total counts/second')
                print(f'{rate_sixty} 60 keV gamma counts/second')
                print(f'{rate_alpha} alpha counts/second w/ err {alp_err}')
                spot.append((rate_alpha, alp_err))
                if r == 12:
                    r12_alp.append((rate_alpha, alp_err))
                if r == 14:
                    if theta == 90:
                        r14_alp.append((rate_alpha, alp_err))
                    if theta == 71:
                        angled_alp.append((rate_alpha, alp_err))

#   plt.figure()
#   plt.errorbar(rotAngle, [r12_alp[i][0] for i in range(len(rotAngle))], [r12_alp[i][1] for i in range(len(rotAngle))], label="r=12", marker='.', ls='none')
#   plt.errorbar(rotAngle, [r14_alp[i][0] for i in range(len(rotAngle))], [r14_alp[i][1] for i in range(len(rotAngle))], label="r=14", marker='.', ls='none')
#   plt.errorbar(rotAngle, [angled_alp[i][0] for i in range(len(rotAngle))], [angled_alp[i][1] for i in range(len(rotAngle))], label="angled", marker='.', ls='none')
#   plt.xlabel("rotary angle [deg]")
#   plt.ylabel("rate [counts/sec]")
#   plt.legend()
#   plt.title("rotary centering scan alpha rates")
#   plt.savefig('rotary_centering_scan_rates.jpg')
#   plt.figure()
#   plt.errorbar(radius, [spot[i][0] for i in range(len(radius))], [spot[i][1] for i in range(len(radius))], label="rot162", marker='.', ls='none')
#   plt.xlabel("radius [mm]")
#   plt.ylabel("rate [counts/sec]")
#   plt.legend()
#   plt.title("spot size scan alpha rates")
#   plt.savefig('spot_size_scan_rates.jpg')

def getCounts(processed_filename):
    df = pd.read_hdf(processed_filename, keys='procdf')
    energy = np.array(df['energy'])
    counts = len(energy)
    return counts

def getCounts_cut(processed_filename, elo, ehi):
    df = pd.read_hdf(processed_filename, keys='procdf')
    energy = np.array(df['energy'])
    cut_df = df.loc[(df.energy > elo) & (df.energy < ehi)]
    cut_energy_keV = np.array(cut_df['energy']*1000)
    counts = len(cut_energy_keV)
    #print(f'{counts} counts in region {elo} to {ehi} keV')

    return(counts)

def getRate(processed_filename, primaries, elo, ehi):
    # see this elog https://elog.legend-exp.org/UWScanner/166
    source_activity = 4.0e4 #40 kBq = 4e4 decays/s
    time_seconds = primaries/(source_activity)
    counts = getCounts_cut(processed_filename, elo, ehi)
    rate = counts/time_seconds #(rate in counts/s)
    rate_err = np.sqrt(counts)/time_seconds

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
