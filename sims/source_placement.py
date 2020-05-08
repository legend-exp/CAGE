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
import math

def main():
    ditch_depth = 2. #depth of the ditch of the PPM ICPC
    rotAxis_toSource_height = 3.5 # height difference from the rotation acis to where the activity is located
    rotAxis_height = 22.5 # height from top of detector to rotation axis, which is (0, 0, 0) in the mother geometry of the simulation


    positionCalc(y_final=14., theta_det=57.)

def positionCalc(y_final, theta_det):
    ditch_depth = 2.
    rotAxis_toSource_height = 3.5 # height difference from the rotation acis to where the activity is located
    rotAxis_height = 22.5 # height from top of detector to rotation axis, which is (0, 0, 0) in the mother geometry of the simulation

    source_yPos = 0.
    source_zPos = rotAxis_toSource_height
    theta_rot = 90.-theta_det
    pi = math.pi
    deg_to_rad = pi/180.

    delta_y_source = rotAxis_toSource_height*(math.cos((90.+theta_rot)*deg_to_rad))
    delta_z_source = rotAxis_toSource_height*(math.sin((90.+theta_rot)*deg_to_rad))

    # print((math.cos((90.+theta_rot)*deg_to_rad)))
    # print(delta_y_source)
    # exit()

    source_zPos = delta_z_source
    real_source_zPos = rotAxis_height + delta_z_source

    if theta_det==90.:
        #y distance along detector surface added from source being rotated. If at norm incidence, y pos of the source is same as y pos on the detector.
        delta_y_rotation = 0.0

    else:
        #y distance along detector surface added from source being rotated. If at norm incidence, y pos of the source is same as y pos on the detector.
        delta_y_rotation = rotAxis_height/(math.tan((theta_det)*deg_to_rad))

    # print(delta_y_rotation)
    # exit()

    if (13.<=y_final<=16.):
        print('Input final y position is located within the ditch.\nOffsetting by ditch depth value: %.1f mm' %ditch_depth)
        delta_y_ditch = ditch_depth/(math.tan((theta_det)*deg_to_rad))
    else:
        delta_y_ditch = 0.0

    if (delta_y_rotation + delta_y_ditch) > y_final:
        print('Y displacement from rotation greater than final y position. Adjusting accordingly.')
        y_diff = y_final - (delta_y_rotation + delta_y_ditch)
        axis_yPos = y_diff
        source_yPos = y_diff + delta_y_source
    else:
        source_yPos = y_final - delta_y_ditch - delta_y_rotation - delta_y_source
        axis_yPos = y_final - delta_y_ditch - delta_y_rotation


    print('For theta_det= %.1f, radius= %.1f:\n source location should be: /gps/pos/centre 0.0 %.3f %.3f mm' %(theta_det, y_final, source_yPos, source_zPos))
    print('Rotation axis of the source in the mother GDML file should be placed at: (0.0 %.3f 0.0 mm)' %axis_yPos)







if __name__ == '__main__':
	main()
