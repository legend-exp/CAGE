#!/usr/bin/env python3
# Written by Gulden Othman September 2020.
# Calculates capacitance from data taking while biasing with a pulser
import numpy as np
import sys
import math
import matplotlib.pyplot as plt

def main():
    # uncertainty_height()
    # uncertainty_angle()
    total()

def uncertainty_height():
    meas1 = [88.8, 89.2, 88.8, 89.0]
    col1 = 88.8
    col2 = 89.2
    col3 = 88.8
    col4 = 89.0

    err1 = 0.3
    err2 = 0.5

    meas2 = [77.8, 78.0, 78.2, 78.2]

    col21 = 77.8
    col22 = 78.0
    col23 = 78.2
    col24 = 78.2

    mean1 = np.mean(meas1)
    mean2 = np.mean(meas2)

    uncert1 = np.sqrt(((err1)**2 + (err1)**2 + (err1)**2+ (err1)**2)/4)
    uncert2 = np.sqrt(((err1)**2 + (err2)**2 + (err1)**2+ (err1)**2)/4)

    var1 = np.var(meas1)
    var2 = np.var(meas2)

    print(f'mean: {mean1}; uncert: {uncert1}; var: {var1}')
    print(f'mean: {mean2}; uncert: {uncert2}; var: {var1}')

    # what delta_theta does that translate to?
    # tan(delta_theta) = hight misalignment/collimator diameter = var1/coll_diameter
    # delta_theta = arctan(var1/coll_diameter)
    coll_diameter = 32 # in mm

    delta_theta = np.arctan(var1/coll_diameter)
    delta_theta_max = np.arctan(uncert2/coll_diameter)

    # tan(delta_theta) = delta_x/height
    # delta_x = height *tan(delta_theta) # linear displacement on detector surface caused by angular misalignment

    height= 22.0 # vertical distance in mm between OPPI surface and rotation axis
    delta_x = height*np.tan(delta_theta)
    delta_x_max = height*np.tan(delta_theta_max)

    print(f'delta_x: {delta_x}; delta_x_max: {delta_x_max}')

    return(delta_x, delta_x_max)

def uncertainty_angle():
    # NEMA 17 motor has 1.8deg/step, 200 steps/revolution, 250 microsteps/step->> 0.0072 deg/microstep

    theta = 0.0072 # in degrees
    theta_rad = theta * (np.pi/180.) # in rad

    height = 22.0 # vertical distance in mm between OPPI surface and rotation axis

    # tan(theta) = delta/height

    delta = height*np.tan(theta_rad)

    print(f'delta: {delta}')

    return(delta)

def total():
    delta1_min, delta1_max = uncertainty_height()
    delta2 = uncertainty_angle()

    # uncertainty from angle rotation, and from 0.5 mm motor resolution
    r_uncert_min = np.sqrt((delta1_min**2) + (delta2**2) + (0.5)**2)
    r_uncert_max = np.sqrt((delta1_max**2) + (delta2**2)+ (0.5)**2)

    print(f'min uncertainty: {r_uncert_min}; max uncertainty: {r_uncert_max}')

    # add in 0.5 mm misalignment from collimator
    r_min = np.sqrt((r_uncert_min**2)+ (0.5)**2)
    r_max = np.sqrt((r_uncert_max**2)+ (0.5)**2)

    print(f'total min uncertainty: {r_min}; total max uncertainty: {r_max}')



if __name__ == '__main__':
	main()
