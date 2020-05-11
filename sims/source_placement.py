#!/usr/bin/env python3
# Written by Gulden Othman May 2020.
# This code tells you where to place the collimator both in simulation and in real life when you would like the source
# center to end up at a specific radius at a specific angle with respect to the detector surface.
# All dimensions in mm, angles in deg
import numpy as np
import sys
import math

def main():

    # y_final is the desired radius on the detector surface where center of beam should aim.
    # theta_det is the desired angle for the source with respect to the detector surface
    # set icpc=Fale if scanning a PPC (or other detector with no ditch)
    # All dimensions in mm, angles in deg

    positionCalc(y_final=6., theta_det=65.)
    # maxRotation(theta_det=45.)

def positionCalc(y_final, theta_det, icpc=True):
    ditch_depth = 2. # ditch depth for ICPC in mm
    rotAxis_toSource_height = 3.5 # height difference in mm from the rotation axis to where the activity is located
    rotAxis_height = 22.5 # height in mm from top of detector to rotation axis, which is (0, 0, 0) in the mother geometry of the simulation

    theta_rot = 90.-theta_det #rotation angle of the collimator with respect to the horizontal. Should be the real-life rotation angle of the source motor
    pi = math.pi
    deg_to_rad = pi/180.

    delta_y_source = rotAxis_toSource_height*(math.cos((90.+theta_rot)*deg_to_rad)) # change in mm of the  y-position within the collimator from source being rotated
    delta_z_source = rotAxis_toSource_height*(math.sin((90.+theta_rot)*deg_to_rad)) # change in mm of the z-position within the collimator from source being rotated

    source_zPos = delta_z_source

    real_source_zPos = rotAxis_height + delta_z_source # Real height in mm of the source relative to the detector surface

    if theta_det==90.:
        # y distance along detector surface added from source being rotated. If at norm incidence, y pos of the source is same as y pos on the detector.
        delta_y_rotation = 0.0

    else:
        # y distance along detector surface added from source being rotated. If at norm incidence, y pos of the source is same as y pos on the detector.
        delta_y_rotation = rotAxis_height/(math.tan((theta_det)*deg_to_rad))

    if icpc==True:
        if (13.<y_final<16.):
            print('Input final y position (radius) is located within the ditch.\nOffsetting by ditch depth value: %.1f mm'   %ditch_depth)
            delta_y_ditch = ditch_depth/(math.tan((theta_det)*deg_to_rad))
        else:
            delta_y_ditch = 0.0

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

    rotUnitVec_y = math.cos(theta_rot*deg_to_rad) # determines the y-coordinate of the rotation vector (/gps/pos/rot2) for rotating the source activity in the g4simple run macro
    rotUnitVec_z = math.sin(theta_rot*deg_to_rad) # determines the z-coordinate of the rotation vector (/gps/pos/rot2) for rotating the source activity in the g4simple run macro

    print('For theta_det= %.1f deg, radius= %.1f mm:' %(theta_det, y_final))
    print('Source location in the run macro should be: \n/gps/pos/centre 0.0 %.3f %.3f mm' %( source_yPos, source_zPos))
    print('Source activity in the run macro should be rotated to: \n/gps/pos/rot1 1 0 0 \n/gps/pos/rot2 0 %.5f %.5f' %(rotUnitVec_y, rotUnitVec_z))

    print('Position of the source ("sourceRotationVolume") in the mother GDML file, should be placed at: \n<position name= "source_center" x="0.0" y="%.3f" z="0.0" unit="mm"/>\n' %axis_yPos)

    print('In the lab, to correspond to theta_det= %.1f deg, at radius= %.1f mm: \nsource motor should be rotated to %.1f deg \nsource should be translated to %.3f mm from center' %(theta_det, y_final, theta_rot, axis_yPos))

def maxRotation(theta_det):
    # Check maximum rotation angle of the collimator in order to leave "min_clearance_toLMFE" mm clearance between the collimator and the LMFE, with the collimator rotation axis "rotAxis_height" mm above the detector surface, and the LMFE "height_det_to_LMFE" mm above the detector surface.
    rad_to_deg = 180./math.pi
    # First check that the source can be rotated to this angle without hitting the LMFE
    rotAxis_height = 22.5 # height in mm from top of detector to rotation axis, which is (0, 0, 0) in the mother geometry of the simulation
    height_det_to_LMFE = 7.0 # height in mm between hieghest point of LMFE and detector surface
    height_LMFE_to_ax = rotAxis_height - height_det_to_LMFE # height in mm between top of LMFE and rotation axis
    min_clearance_toLMFE = 4 # minimum height in mm to maintain of collimator above LMFE
    height_ax_to_bottom = height_LMFE_to_ax - min_clearance_toLMFE # Height between bottom edge of collimator anrotation axis
    collHeight_ax_to_top = 6 # mm
    collHeight_ax_to_bottom = 3 # mm
    collRadius = 16 # mm
    collMaxClearance = math.sqrt(collHeight_ax_to_top**2 +collRadius**2)
    coll_second_MaxClearance = math.sqrt(collHeight_ax_to_bottom**2 +collRadius**2)
    theta_bottom_to_second_max = math.atan(collHeight_ax_to_bottom/collRadius)

    theta_rot_max = (math.asin(height_ax_to_bottom/coll_second_MaxClearance) - theta_bottom_to_second_max)*rad_to_deg
    theta_det_min = 90 - theta_rot_max
    print('Maximum rotation angle for %.1f mm clearance above LMFE: %.2f deg' %(min_clearance_toLMFE, theta_rot_max))
    print('Corresponding angle with respect to detector surface: %.2f' %theta_det_min)
    print(height_ax_to_bottom)

def checkRotation(theta_det):
    #THIS IS NOT WORKING YET!!!
    # First check that the source can be rotated to this angle while still maintaining desired LMFE clearance
    # Check maximum rotation angle of the collimator in order to leave "min_clearance_toLMFE" mm clearance between the collimator and the LMFE, with the collimator rotation axis "rotAxis_height" mm above the detector surface, and the LMFE "height_det_to_LMFE" mm above the detector surface.
    print('this is not working yet!')
    exit()
    pi = math.pi
    theta_rot = 90.-theta_det
    rad_to_deg = 180./pi
    deg_to_rad = pi/180.

    rotAxis_height = 22.5 # height in mm from top of detector to rotation axis, which is (0, 0, 0) in the mother geometry of the simulation
    height_det_to_LMFE = 7.0 # height in mm between hieghest point of LMFE and detector surface
    height_LMFE_to_ax = rotAxis_height - height_det_to_LMFE # height in mm between top of LMFE and rotation axis
    min_clearance_toLMFE = 4 # minimum height in mm to maintain of collimator above LMFE
    height_ax_to_bottom = height_LMFE_to_ax - min_clearance_toLMFE # Height between bottom edge of collimator and rotation axis
    collHeight_ax_to_top = 6 # mm
    collHeight_ax_to_bottom = 3 # mm
    collRadius = 16 # mm
    collMaxClearance = math.sqrt(collHeight_ax_to_top**2 +collRadius**2)
    coll_second_MaxClearance = math.sqrt(collHeight_ax_to_bottom**2 +collRadius**2)
    theta_bottom_to_second_max = math.atan(collHeight_ax_to_bottom/collRadius)

    theta_rot_max = (math.asin(height_ax_to_bottom/coll_second_MaxClearance) - theta_bottom_to_second_max)*rad_to_deg

    theta_det_min = 90 - theta_rot_max

    # print(coll_second_MaxClearance)
    # print(math.sin(theta_bottom_to_second_max+theta_rot))
    # print(coll_second_MaxClearance*math.sin((theta_bottom_to_second_max+(theta_rot)*deg_to_rad)*deg_to_rad))
    # exit()

    if height_ax_to_bottom <= coll_second_MaxClearance*math.sin((theta_bottom_to_second_max+(theta_rot)*deg_to_rad)*deg_to_rad):
        print('Rotation fulfills %.1f mm min clearance above LMFE' %min_clearance_toLMFE)
        print(height_ax_to_bottom)
        return(True)
    else:
        print('Rotation does not maintain %.1f mm min clearance above LMFE!!' %min_clearance_toLMFE)
        print('Maximum rotation angle for %.1f mm clearance above LMFE: %.2f deg' %(min_clearance_toLMFE, theta_rot_max))
        print('Corresponding angle with respect to detector surface: %.2f' %theta_det_min)
        return(False)

if __name__ == '__main__':
	main()
