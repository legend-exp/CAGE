#!/usr/bin/env python3
# Written by Gulden Othman May 2020.
# This code tells you where to place the collimator both in simulation and in real life when you would like the source
# center to end up at a specific radius at a specific angle with respect to the detector surface.
# All dimensions in mm, angles in deg
# This works only for the ICPC detecor and lead collimator in this elog: https://elog.legend-exp.org/UWScanner/180
# Other relevant elogs and useful diagrams: https://elog.legend-exp.org/UWScanner/182; https://elog.legend-exp.org/UWScanner/181
import numpy as np
import sys
import math

def main():

    # y_final is the desired radius on the detector surface where center of beam should aim.
    # theta_det is the desired angle for the source with respect to the detector surface
    # set icpc=False if scanning a PPC (or other detector with no ditch)-- But note, all other dimensions must be updated first for the output to be correct!!
    # All dimensions in mm, angles in deg

    positionCalc(y_final=6., theta_det=45.)
    # maxRotation(min_clearance_toLMFE=5.0)
    # checkRotation(theta_det=45., min_clearance_toLMFE=5.0)
    # thetaCalc(y_final=14., icpc=True)

def positionCalc(y_final, theta_det, icpc=True):
    theta_rot = 90.-theta_det #rotation angle of the collimator with respect to the horizontal. Should be the real-life rotation angle of the source motor
    pi = math.pi
    deg_to_rad = pi/180.

    # First check that it's safe to rotate to this angle
    rotCheck = checkRotation(theta_det)
    # if rotCheck[0]==False:
        # exit()

    ditch_depth = 2. # ditch depth for ICPC in mm
    rotAxis_toSource_height = 3.5 # height difference in mm from the rotation axis to where the activity is located
    rotAxis_height = 22.5 # height in mm from top of detector to rotation axis, which is (0, 0, 0) in the mother geometry of the simulation

    delta_y_source = rotAxis_toSource_height*(math.cos((90.+theta_rot)*deg_to_rad)) # change in mm of the y-position of the source activity within the collimator from source being rotated
    delta_z_source = rotAxis_toSource_height*(math.sin((90.+theta_rot)*deg_to_rad)) # change in mm of the z-position of the source activity within the collimator from source being rotated

    source_zPos = delta_z_source

    real_source_zPos = rotAxis_height + delta_z_source # Real height in mm of the source relative to the detector surface

    # Do this to avoid a divide by zero problem
    if theta_det==90.:
        # y distance along detector surface added from source being rotated. If at norm incidence, y pos of the source is same as y pos on the detector.
        delta_y_rotation = 0.0

    else:
        # y distance along detector surface added from source being rotated. If at norm incidence, y pos of the source is same as y pos on the detector.
        delta_y_rotation = rotAxis_height/(math.tan((theta_det)*deg_to_rad))

    if icpc==True:
        if (13.<y_final<16.):
            print('Input final y position (radius) is located within the ditch.\nOffsetting by ditch depth value: %.1f mm \n'   %ditch_depth)
            delta_y_ditch = ditch_depth/(math.tan((theta_det*deg_to_rad)))
        else:
            delta_y_ditch = 0.0

    else:
        delta_y_ditch = 0.0

    # print('delta_y_ditch: ', delta_y_ditch)

    delta_y_tot = delta_y_rotation + delta_y_ditch # total y displacement, wrt the rotation axis, from rotating the source and from final desired y position (y_final) being in the ditch if it is
    axis_yPos = y_final - delta_y_tot # final y-position the axis of the collimator (y-coord of center of "sourceRotationVolume")
    source_yPos = axis_yPos - np.abs(delta_y_source) # final y-position of the source activity within the collimator (specified in g4simple run macro)

    rotUnitVec_y = math.cos(theta_rot*deg_to_rad) # determines the y-coordinate of the rotation vector (/gps/pos/rot2) for rotating the source activity in the g4simple run macro
    rotUnitVec_z = math.sin(theta_rot*deg_to_rad) # determines the z-coordinate of the rotation vector (/gps/pos/rot2) for rotating the source activity in the g4simple run macro

    print('For theta_det= %.1f deg, radius= %.1f mm:' %(theta_det, y_final))
    print('Source activity in the run macro should be rotated to and centered according to: \n/gps/pos/rot1 1 0 0 \n/gps/pos/rot2 0 %.5f %.5f' %(rotUnitVec_y, rotUnitVec_z))
    print('/gps/pos/centre 0.0 %.3f %.3f mm \n' %( source_yPos, source_zPos))


    print('Position of the source ("sourceRotationVolume") in the mother GDML file, should be placed at: \n<position name= "source_center" x="0.0" y="%.3f" z="0.0" unit="mm"/> \n<rotation name="source%.0f" x="-%.2f" unit="deg"/> \n' %(axis_yPos, theta_rot, theta_rot))

    print('In the lab, to correspond to theta_det= %.1f deg, at radius= %.1f mm: \nsource motor should be rotated to %.1f deg \nsource should be translated to %.3f mm from center' %(theta_det, y_final, theta_rot, axis_yPos))

def thetaCalc(y_final, icpc=True):
    # Caluclate the rotation angle to rotate the source WHILE KEEPING IT CENTERED OVER P+ CONTACT to reach desired "y_final" radius in mm on detector surface
    ditch_depth = 2. # ditch depth for ICPC in mm
    rotAxis_toSource_height = 3.5 # height difference in mm from the rotation axis to where the activity is located
    rotAxis_height = 22.5 # height in mm from top of detector to rotation axis, which is (0, 0, 0) in the mother geometry of the simulation
    pi = math.pi
    deg_to_rad = pi/180.
    rad_to_deg = 180./pi

    if (icpc==True and (13.<y_final<16.)):
        print('Input final y position (radius) is located within the ditch.\nOffsetting by ditch depth value: %.1f mm'   %ditch_depth)
        rotAxis_height += ditch_depth

    theta_rot = math.atan(y_final/rotAxis_height)*rad_to_deg #rotation angle of the collimator with respect to the horizontal. Should be the real-life rotation angle of the source motor

    theta_det = 90.-theta_rot #angle of the beam with respect to the detector surface


    delta_y_source = rotAxis_toSource_height*(math.cos((90.+theta_rot)*deg_to_rad)) # change in mm of the y-position of the source activity within the collimator from source being rotated
    delta_z_source = rotAxis_toSource_height*(math.sin((90.+theta_rot)*deg_to_rad)) # change in mm of the z-position of the source activity within the collimator from source being rotated

    source_yPos = delta_y_source
    source_zPos = delta_z_source

    rotUnitVec_y = math.cos(theta_rot*deg_to_rad) # determines the y-coordinate of the rotation vector (/gps/pos/rot2) for rotating the source activity in the g4simple run macro
    rotUnitVec_z = math.sin(theta_rot*deg_to_rad) # determines the z-coordinate of the rotation vector (/gps/pos/rot2) for rotating the source activity in the g4simple run macro

    print('To reach radius %.1f mm while keeping the collimator centered above p+ contact, rotate collimator to: %.2f deg' %(y_final, theta_rot))
    print('theta_rot = %.2f deg corresponds to theta_det = %.2f \n' %(theta_rot, theta_det))

    print('For the simulations: ')
    print('Source activity in the run macro should be rotated to and centered according to: \n/gps/pos/rot1 1 0 0 \n/gps/pos/rot2 0 %.5f %.5f' %(rotUnitVec_y, rotUnitVec_z))
    print('/gps/pos/centre 0.0 %.3f %.3f mm \n' %( source_yPos, source_zPos))
    print('Position of the source ("sourceRotationVolume") in the mother GDML file, should be placed at: \n<position name= "source_center" x="0.0" y="0.0" z="0.0" unit="mm"/> \n<rotation name="source%.0f" x="-%.2f" unit="deg"/>' %(theta_rot, theta_rot))


def maxRotation(min_clearance_toLMFE=5.0):
    # Check maximum rotation angle of the collimator in order to leave "min_clearance_toLMFE" mm clearance between the collimator and the LMFE, with the collimator rotation axis "rotAxis_height" mm above the detector surface, and the LMFE "height_det_to_LMFE" mm above the detector surface.
    rad_to_deg = 180./math.pi
    deg_to_rad = math.pi/180.
    rotAxis_height = 22.5 # height in mm from top of detector to rotation axis, which is (0, 0, 0) in the mother geometry of the simulation
    height_det_to_LMFE = 7.0 # height in mm between hieghest point of LMFE and detector surface
    height_LMFE_to_ax = rotAxis_height - height_det_to_LMFE # height in mm between top of LMFE and rotation axis
    #min_clearance_toLMFE = 5. # minimum height in mm to maintain of collimator above LMFE
    collRadius = 16 # mm

    theta_rot_max = math.asin((height_LMFE_to_ax-min_clearance_toLMFE)/collRadius)*rad_to_deg
    theta_det_min = 90 - theta_rot_max

    print('Maximum rotation angle of the collimator to maintain %.2f mm clearance between the bottom edge of the (attenuator part of the) collimator and LMFE: \n%.2f deg' %(min_clearance_toLMFE, theta_rot_max))
    print('Corresponds to minimum angle with respect to the detector of: %.2f deg' %(theta_det_min))

    return(theta_rot_max, theta_det_min)

def checkRotation(theta_det, min_clearance_toLMFE=5.0):
    # First check that the source can be rotated to this angle while still maintaining desired LMFE clearance
    # Check maximum rotation angle of the collimator in order to leave "min_clearance_toLMFE" mm clearance between the collimator and the LMFE, with the collimator rotation axis "rotAxis_height" mm above the detector surface, and the LMFE "height_det_to_LMFE" mm above the detector surface.

    rad_to_deg = 180./math.pi
    deg_to_rad = math.pi/180.
    theta_rot = (90 - theta_det)
    rotAxis_height = 22.5 # height in mm from top of detector to rotation axis, which is (0, 0, 0) in the mother geometry of the simulation
    height_det_to_LMFE = 7.0 # height in mm between hieghest point of LMFE and detector surface
    height_LMFE_to_ax = rotAxis_height - height_det_to_LMFE # height in mm between top of LMFE and rotation axis
    #min_clearance_toLMFE = 5. # minimum height in mm to maintain of collimator above LMFE

    collRadius = 16 # mm
    delta_z = collRadius*math.sin(theta_rot*deg_to_rad) # z-distance in mm the bottom edge of the (attenuator part of the) collimator moves downward due to rotation
    z_final = rotAxis_height - delta_z # final z-distance in mm between the top of the detector and the bottom edge of the (attenuator part of the) collimator
    z_lmfe = height_LMFE_to_ax - delta_z

    if z_lmfe < min_clearance_toLMFE:
        print('The rotation angle theta_det= %.1f (theta_rot = %.1f) does not maintain the maximum clearance of %.2f mm between LMFE and lowest edge of collimator when top-hat is down! \nActual clearance: %.2f mm' %(theta_det, theta_rot, min_clearance_toLMFE, z_lmfe))
        # print('theta_rot must be less than %.2f deg \ntheta_det must be more than %.2f deg' %(maxRotation(min_clearance_toLMFE=5.0)[0], maxRotation(min_clearance_toLMFE=5.0)[1]))
        return(False, min_clearance_toLMFE, z_lmfe)
    else:
        print('The rotation angle theta_det= %.1f (theta_rot = %.1f) safely maintains the maximum clearance of %.2f mm between LMFE and lowest edge of collimator when top-hat is down! \nActual clearance: %.2f mm' %(theta_det, theta_rot, min_clearance_toLMFE, z_lmfe))
        return(True, min_clearance_toLMFE, z_lmfe)

if __name__ == '__main__':
	main()
