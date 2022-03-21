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
    radius = [15]
    thetaDet = [45, 60, 75, 90]
    rotAngle = [0]
    scan = 'source_angle_scan'
    plot = True

    rates = scanRate(scan, radius, thetaDet, rotAngle)
    print(rates)
    if plot:
        x_ax = 'thetaDet' # 'radius', 'thetaDet', OR 'rotary'
        lines = 'radius' # same choices
        name = f'{scan}_rates'
        plotRates(rates, x_ax, lines, name) 
#   radius = [5, 6, 7, 8, 9, 10]
#   thetaDet = [90]
#   rotAngle = [162]
#   scan = 'spot_size_scan'
    

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
def scanRate(scan, radius, thetaDet, rotAngle):
    processed_dir = f'../data/oppi/{scan}'
    primaries = 1e8
    source_activity = 4e4 #40 kBq = 4e4 decays/s
    activity_err = 4e4*0.3 # +- 30%
    time_seconds = primaries/(source_activity)
    rate_arr = []
    rates = pd.DataFrame(columns=['radius', 'thetaDet', 'rotary', 'rate', 'rate_err']) 

    for r in radius:
        for theta in thetaDet:
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
                entry = [r, theta, rot, rate_alpha, alp_err]
                rate_arr.append(entry)
                print(name + ':')
                print(f'{rate} total counts/second')
                print(f'{rate_sixty} 60 keV gamma counts/second')
                print(f'{rate_alpha} alpha counts/second w/ err {alp_err}')
    rate_arr = np.array(rate_arr)
    rates = pd.DataFrame(data=rate_arr, columns=['radius', 'thetaDet', 'rotary', 'rate', 'rate_err'])
    return rates

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

def plotRates(rates, x_ax, lines, name): 
    if x_ax == lines:
        print('Cannot have x_ax == lines')
        return

    axis = np.sort(rates[x_ax].drop_duplicates().to_list())
    l = np.sort(rates[lines].drop_duplicates().to_list())

    plt.figure(figsize=(12,8))
    plt.xlabel(x_ax)
    plt.ylabel("Alpha Rate (cts/sec)")
    plt.title(name)
    for line in l:
        dl = rates.loc[rates[lines] == line].sort_values(x_ax)
        data = dl['rate']
        data_err = dl['rate_err']
        plt.errorbar(axis, data, data_err, label=line, marker='.', ls='none')
    plt.legend()
    plt.savefig(f'{name}.jpg')


if __name__ == '__main__':
	main()
