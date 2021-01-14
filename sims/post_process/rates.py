import numpy as np
import scipy
import matplotlib
from matplotlib.colors import LogNorm
from scipy.stats import norm, kde
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import ROOT
import sys
from particle import PDGID
matplotlib.rcParams['text.usetex'] = True

def main():
    base_filename = '../alpha/processed_out/oppi/processed'
    processed_filename = '../alpha/processed_out/oppi/processed_oppi_ring_y5_norm_241Am_100000000.hdf5'

    primaries = 10000000
    radius = [5, 6, 7, 8, 10] # in mm
    elo = 5.4 # in MeV
    ehi = 5.6 # in MeV

    getCounts(processed_filename) # get all counts in physical volume for this file. Useful for debugging if sim was successful
    # getCounts_cut(processed_filename, elo, ehi) # get counts within specific energy region
    # getRate(processed_filename, primaries, elo, ehi) # get rate in counts/sec for specific energy region
    # plotRate(radius, elo, ehi) # plot rates for multiple source positions (sims files) on one plot


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
	print(f'{rate} counts/second in region {elo} to {ehi} keV')

def plotRate(radius, elo, ehi):
	rates_arr = []
	for r in radius:
		rate = getRate(f'../alpha/processed_out/oppi/processed_oppi_ring_y{r}_norm_241Am_100000000.hdf5', 10000000, 5.4, 5.6)
		rates_arr.append(rate)
	plt.plot(radius, rates_arr, '.r')
	plt.show()

	return(rate)

if __name__ == '__main__':
	main()
