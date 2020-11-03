#!/usr/bin/env python3
# Written by Gulden Othman September 2020.
# Calculates capacitance from data taking while biasing with a pulser
import numpy as np
import sys
import math
import matplotlib.pyplot as plt

def main():
    oppi()

def oppi():
    c_f = 0.26 #pF
    biasV_m1 = np.array([1000, 1500, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500])
    baseline_m1 = np.array([-808, -808, -888, -888, -897, -888, -922, -922, -922, -923])
    V_in_m1 = np.array([200, 200, 200, 200, 200, 200, 200, 200, 200, 200])/11.
    V_1_m1 = np.array([2320, 1440, 880, 760, 600, 440, 92, 148, 136, 140])
    V_2_m1 = np.array([2240, 1520, 880, 740, 600, 420, 240, 240, 240, 240])

    # values from elog 224: https://elog.legend-exp.org/UWScanner/224
    # V_in: pulser value (200 mVpp; HV filterbox has a divide by 11 attenuation of the pulse voltage from the waveform generator)
    # V_1: first stage Vpp
    biasV_m2 = np.array([2000, 2050, 2100, 2150, 2200, 2250, 2300, 2350, 2400, 2450, 2500, 2600, 2700, 2800, 2900, 3000, 3100]) # modified to include both HV scans from this day
    biasV_m2_1 = np.array([2500, 2550,  2600, 2650, 2700, 2750, 2800, 2850, 2900, 2950, 3000, 3050, 3100]) # only first HV scan, BL measurements taken with pulser on
    biasV_m2_2 = np.array([2000, 2050, 2100, 2150, 2200, 2250, 2300, 2350, 2400, 2450, 2500, 2600, 2700, 2800]) # only second HV scan, BL measurements taken with pulser off
    baseline_m2_1 = np.array([-879, -880, -880, -880, -880, -880, -880, -884, -895, -915, -950, -1000, -1300]) # only first HV scan, BL measurements taken with pulser on
    baseline_m2_2 = np.array([-808, -805, -806, -800, -802, -803, -797, -794, -795, -796, -798, -798, -798, -810]) #
    V_in_m2 = np.array([200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200])/11.
    V_1_m2 = np.array([528, 448, 360, 232, 148, 148, 144, 140, 140, 138, 138, 140, 140, 140, 144, 144, 144])

    cap_vs_V(c_f, biasV_m2, V_1_m2, V_in_m2, plot=True)
    baseline_vs_V(biasV_m2_2, baseline_m2_2, plot=True)
    # baseline_vs_V(biasV_m2_1, baseline_m2_1, plot=True)

def cap_vs_V(c_f, biasV, V_1, V_in, plot=True):
    cap = np.zeros(len(V_1))
    for i in range(len(V_1)):
        cap[i] = c_f*(V_1[i]/(V_in[i]))
    if plot:
        fig = plt.figure()
        plt.plot(biasV, cap, '.r')
        plt.title('OPPI Capacitance vs Voltage')
        plt.xlabel('Voltage (V)')
        plt.ylabel('Capacitance (pF)')
        # plt.yscale('log')
        plt.show()
    return(cap)

def baseline_vs_V(biasV, baseline, plot=True):
    fig = plt.figure()
    plt.plot(biasV, baseline, '.r')
    plt.title('OPPI Baseline vs Voltage')
    # plt.title('OPPI Baseline vs Voltage\n (taken with pulser on)') # if using first, high-bias V scan from elog 224
    plt.xlabel('Voltage (V)')
    plt.ylabel('Average Baseline (mV)')
    # plt.ylim(-815, -789)
    plt.show()


if __name__ == '__main__':
	main()
