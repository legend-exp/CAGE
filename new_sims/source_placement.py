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

    #calculate_CollClearances()

    # positionCalc(y_final=14, theta_det=45, icpc=True)
    # rotaryCalc(radius=12.0, d_theta=10)
    maxRotation(min_clearance_toLMFE=5.0, icpc=True)
    checkRotation(theta_det=65, min_clearance_toLMFE=5.0, icpc=True)
    # thetaCalc(y_final=12., icpc=False)

def positionCalc(y_final, theta_det, icpc=False):
    theta_rot = 90.-theta_det #rotation angle of the collimator with respect to the horizontal. Used in calculations, where 0 deg theta_rot is normal incidence on the detector surface
    theta_rot_motor = theta_rot - 180. #rotation angle of the collimator with respect to the horizontal. Should be the real-life rotation angle of the source motor, where -180 deg is normal incidence on the detector surface
    pi = math.pi
    deg_to_rad = pi/180.

    # First check that it's safe to rotate to this angle
    rotCheck = checkRotation(theta_det, icpc=icpc)
    # if rotCheck[0]==False:
        # exit()

    ditch_depth = 2. # ditch depth for ICPC in mm
    rotAxis_toSource_height = 3.85 # height difference in mm from the rotation axis to where the activity is located (updated for collimator v3)
    if icpc==False:
        rotAxis_height = 23.8 # height for OPPI in mm from top of detector to rotation axis, which is (0, 0, 0) in the mother geometry of the simulation
        print('Using OPPI axis height: % .1f' %rotAxis_height)
    else:
        rotAxis_height = 25.1 # height in mm from top of detector to rotation axis, which is (0, 0, 0) in the mother geometry of the simulation
        print('Using ICPC axis height: % .1f' %rotAxis_height)

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
        if (13<y_final<16.):
            print('Input final y position (radius) is located within the ditch.\nOffsetting by ditch depth value: %.1f mm \n'   %ditch_depth)
            delta_y_ditch = ditch_depth/(math.tan((theta_det*deg_to_rad)))
        else:
            delta_y_ditch = 0.0

    else:
        delta_y_ditch = 0.0

    # print('delta_y_ditch: ', delta_y_ditch)

    delta_y_tot = delta_y_rotation + delta_y_ditch # total y displacement, wrt the rotation axis, from rotating the source and from final desired y position (y_final) being in the ditch if it is
    axis_yPos = y_final - delta_y_tot # final y-position the axis of the collimator (y-coord of center of "sourceRotationVolume")
    lab_axis_yPos = axis_yPos
    source_yPos = (axis_yPos - np.abs(delta_y_source)) # final y-position of the source activity within the collimator (specified in g4simple run macro)

    rotUnitVec_y = math.cos(theta_rot*deg_to_rad) # determines the y-coordinate of the rotation vector (/gps/pos/rot2) for rotating the source activity in the g4simple run macro
    rotUnitVec_z = math.sin(theta_rot*deg_to_rad) # determines the z-coordinate of the rotation vector (/gps/pos/rot2) for rotating the source activity in the g4simple run macro

    dirUnitVec_y = math.sin(theta_rot*deg_to_rad)
    dirUnitVec_z = -math.cos(theta_rot*deg_to_rad)
    if axis_yPos < 0.:
        rotary_motor_theta = -180.
        source_rot_motor = (180.+theta_rot)
        lab_axis_yPos *= -1 # this needs to be positive, since will be driving the linear stage "forward" at -180 deg, but its equivalent to a negative axis_yPos

    else:
        rotary_motor_theta = 0.
        source_rot_motor = (theta_rot)

    macro_rotation = f'/gps/pos/rot1 0 1 0 \n/gps/pos/rot2 {rotUnitVec_y:.5f} 0 {rotUnitVec_z:.5f} \n/gps/pos/centre {source_yPos:.3f} 0.0 {source_zPos:.3f} mm \n'
    # macro_rotation = f'/gps/pos/rot1 0 1 0 \n/gps/pos/rot2 {rotUnitVec_y:.5f} 0 {rotUnitVec_z:.5f} \n/gps/pos/centre {source_yPos:.3f} 0.0 {source_zPos:.3f} mm \n/gps/direction 0 {dirUnitVec_y:.5f} {dirUnitVec_z:.5f}'
    gdml_source_center = f'     <position name= "source_center" x="{axis_yPos:.3f}" y="0.0" z="0" unit="mm"/>\n'
    gdml_source_rotation = f'   <rotation name="source{theta_rot:.0f}" y="{theta_rot:.2f}" unit="deg"/> \n'


    print('For theta_det= %.1f deg, radius= %.1f mm:' %(theta_det, y_final))
    print(f'Source activity in the run macro should be rotated to and centered according to: \n{macro_rotation}')


    print(f'Position of the source ("sourceRotationVolume") in the mother GDML file, should be placed at: \n {gdml_source_center} {gdml_source_rotation}')

    print(f'In the lab, to correspond to theta_det= {theta_det:.1f} deg, at radius= {y_final:.1f} mm: \nrotary motor should be rotated to {rotary_motor_theta:.1f} deg \nsource should be translated to {lab_axis_yPos:.3f} mm from center \nsource motor should be rotated to {source_rot_motor:.1f} deg ')

    return macro_rotation, gdml_source_center, gdml_source_rotation


def thetaCalc(y_final, icpc=False):
    # Caluclate the rotation angle to rotate the source WHILE KEEPING IT CENTERED OVER P+ CONTACT to reach desired "y_final" radius in mm on detector surface
    ditch_depth = 1.95 # ditch depth for ICPC in mm
    rotAxis_toSource_height = 3.85 # height difference in mm from the rotation axis to where the activity is located (updated for collimator v3)
    # rotAxis_height = 22.5 # height in mm from top of detector to rotation axis, which is (0, 0, 0) in the mother geometry of the simulation
    pi = math.pi
    deg_to_rad = pi/180.
    rad_to_deg = 180./pi

    if icpc==False:
        rotAxis_height = 23.8 # height for OPPI in mm from top of detector to rotation axis, which is (0, 0, 0) in the mother geometry of the simulation
        print('Using OPPI axis height: % .1f' %rotAxis_height)
    else:
        rotAxis_height = 25.1 # height in mm from top of detector to rotation axis, which is (0, 0, 0) in the mother geometry of the simulation
        print('Using ICPC axis height: % .1f' %rotAxis_height)

    if (icpc==True and (12.9<y_final<15.8)):
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

def rotaryCalc(radius=1.0, d_theta = 2.):
    rad_to_deg = 180/np.pi
    deg_to_rad = np.pi/180
    board_width = 12.76 # in mm
    theta_tot = (2*np.arcsin((board_width/2)/radius))*rad_to_deg
    d_s = (d_theta*deg_to_rad)*radius

    if d_theta >= 10:
        theta_min = 72.08-(theta_tot)/2-2*d_theta
        theta_max = 72.08+(theta_tot)/2+2*d_theta
    else:
        theta_min = 72.08-(theta_tot)/2-20
        theta_max = 72.08+(theta_tot)/2+20



    scan_points = np.arange(theta_min, theta_max, d_theta)
    print(f'for r={radius}: \ntheta_tot = {theta_tot} \nd_theta = {d_theta} \ntheta_min = {theta_min} \ntheta_max = {theta_max} \nscan points = {scan_points}')




def maxRotation(min_clearance_toLMFE=5.0, icpc=False):
    # Check maximum rotation angle of the collimator in order to leave "min_clearance_toLMFE" mm clearance between the collimator and the LMFE, with the collimator rotation axis "rotAxis_height" mm above the detector surface, and the LMFE "height_det_to_LMFE" mm above the detector surface.
    rad_to_deg = 180./math.pi
    deg_to_rad = math.pi/180.
    if icpc==True:
        rotAxis_height = 25.1 # height in mm from top of detector to rotation axis, which is (0, 0, 0) in the mother geometry of the simulation
        height_det_to_LMFE = 7.95 # height in mm between hieghest point of LMFE and detector surface 
        print('Calculating maximum rotation angle for ICPC')
    else:
        rotAxis_height = 23.8 # height in mm from top of detector to rotation axis, which is (0, 0, 0) in the mother geometry of the simulation
        height_det_to_LMFE = 7.89 # height in mm between hieghest point of LMFE and detector surface. Jason estimated 7.89 ± 0.26 mm in elog 324
        print('Calculating maximum rotation angle for OPPI')

    height_LMFE_to_ax = rotAxis_height - height_det_to_LMFE # height in mm between top of LMFE and rotation axis
#    coll_Radius_solder =  15.04 # mm
#    coll_solder_to_ax = 0.125 # mm
#    coll_eff_Radius_solder = np.sqrt(coll_Radius_solder**2+coll_solder_to_ax**2) # in mm. since G10 shaft, hence rotation axis, is actually about 0.125 mm below part soldered onto collimator, get the hypotenuse for "effective radius"
    coll_Radius =  31.6/2 # mm
    coll_to_ax = 1 # mm
    coll_eff_Radius = np.sqrt(coll_Radius**2+coll_to_ax**2) # in mm. since G10 shaft, hence rotation axis, is actually about (1.75 + 0.125) mm below lower part of "attenuator" part of collimator, get the hypotenuse for "effective radius"

#    theta_i_solder = math.atan(coll_solder_to_ax/coll_Radius_solder)*rad_to_deg 
    theta_i = math.atan(coll_to_ax/coll_Radius)*rad_to_deg
    theta_rot_max = math.asin((height_LMFE_to_ax-min_clearance_toLMFE)/coll_eff_Radius)*rad_to_deg - theta_i

    print("Initial theta: ", theta_i)

#    theta_rot_max_solder = math.asin((height_LMFE_to_ax-min_clearance_toLMFE)/coll_eff_Radius_solder)*rad_to_deg +theta_i_solder

#    if theta_rot_max < theta_rot_max_solder:
#        print('Limiting dimension is from rotation center to full collimator width')
#    else:
#        print('Limiting dimension is from rotation center to piece soldered onto collimator')
#        theta_rot_max = theta_rot_max_solder

    theta_det_min = 90 - theta_rot_max

    print('Maximum rotation angle of the collimator to maintain %.2f mm clearance between the bottom edge of the (attenuator part of the) collimator and LMFE: \n%.2f deg' %(min_clearance_toLMFE, theta_rot_max))
    print('Corresponds to minimum angle with respect to the detector of: %.2f deg' %(theta_det_min))

    return(theta_rot_max, theta_det_min)

def checkRotation(theta_det, min_clearance_toLMFE=5.0, icpc=False):
    # First check that the source can be rotated to this angle while still maintaining desired LMFE clearance
    # Check maximum rotation angle of the collimator in order to leave "min_clearance_toLMFE" mm clearance between the collimator and the LMFE, with the collimator rotation axis "rotAxis_height" mm above the detector surface, and the LMFE "height_det_to_LMFE" mm above the detector surface.

    rad_to_deg = 180./math.pi
    deg_to_rad = math.pi/180.
    theta_rot = (90 - theta_det)

    if icpc==True:
        rotAxis_height = 25.1 # height in mm from top of detector to rotation axis, which is (0, 0, 0) in the mother geometry of the simulation
        height_det_to_LMFE = 7.95 # height in mm between hieghest point of LMFE and detector surface
        print('Calculating maximum ratoation angle for ICPC')
    else:
        rotAxis_height = 23.8 # height in mm from top of detector to rotation axis, which is (0, 0, 0) in the mother geometry of the simulation
        height_det_to_LMFE = 7.89 # height in mm between hieghest point of LMFE and detector surface. Jason estimated 7.89 ± 0.26 mm in elog 324
        print('Calculating maximum rotation angle for OPPI')

    height_LMFE_to_ax = rotAxis_height - height_det_to_LMFE # height in mm between top of LMFE and rotation axis
    #min_clearance_toLMFE = 5. # minimum height in mm to maintain of collimator above LMFE

#    coll_Radius_solder =  15.04 # mm
#    coll_solder_to_ax = 0.125 # mm
#    coll_eff_Radius_solder = np.sqrt(coll_Radius_solder**2+coll_solder_to_ax**2) # in mm. since G10 shaft, hence rotation axis, is actually about 0.125 mm below part soldered onto collimator, get the hypotenuse for "effective radius"
    coll_Radius =  31.75/2 # mm
    coll_to_ax = 1.75 + 0.125 # mm
    coll_eff_Radius = np.sqrt(coll_Radius**2+coll_to_ax**2) # in mm. since G10 shaft, hence rotation axis, is actually about (1.75 + 0.125) mm below lower part of "attenuator" part of collimator, get the hypotenuse for "effective radius"

#    theta_i_solder = math.atan(coll_solder_to_ax/coll_Radius_solder)*rad_to_deg 
    theta_i = math.atan(coll_to_ax/coll_Radius)*rad_to_deg

    delta_z = coll_eff_Radius*math.sin(theta_rot*deg_to_rad - theta_i*deg_to_rad) # z-distance in mm the bottom edge of the (attenuator part of the) collimator moves downward due to rotation
    z_final = rotAxis_height - delta_z # final z-distance in mm between the top of the detector and the bottom edge of the (attenuator part of the) collimator
    z_lmfe = height_LMFE_to_ax - delta_z
#    delta_z_solder = coll_eff_Radius_solder*math.sin(theta_rot*deg_to_rad - theta_i_solder*deg_to_rad) # z-distance in mm the bottom edge of the (attenuator part of the) collimator moves downward due to rotation
#    z_final_solder = rotAxis_height - delta_z_solder # final z-distance in mm between the top of the detector and the bottom edge of the (attenuator part of the) collimator
#    z_lmfe_solder = height_LMFE_to_ax - delta_z_solder

#    if z_lmfe > z_lmfe_solder:
#        print('Limiting portion is the piece soldered onto the collimator')
#        z_lmfe = z_lmfe_solder

    if z_lmfe < min_clearance_toLMFE:
        print('The rotation angle theta_det= %.1f (theta_rot = %.1f) DOES NOT maintain the maximum clearance of %.2f mm between LMFE and lowest edge of collimator when top-hat is down! \nActual clearance: %.2f mm' %(theta_det, theta_rot, min_clearance_toLMFE, z_lmfe))
        # print('theta_rot must be less than %.2f deg \ntheta_det must be more than %.2f deg' %(maxRotation(min_clearance_toLMFE=5.0)[0], maxRotation(min_clearance_toLMFE=5.0)[1]))
        return(False, min_clearance_toLMFE, z_lmfe)
    else:
        print('The rotation angle theta_det= %.1f (theta_rot = %.1f) safely maintains the maximum clearance of %.2f mm between LMFE and lowest edge of collimator when top-hat is down! \nActual clearance: %.2f mm' %(theta_det, theta_rot, min_clearance_toLMFE, z_lmfe))
        return(True, min_clearance_toLMFE, z_lmfe)


def calculate_CollClearances():
    # Longest distances between G10 shaft and various points of collimator.
    # Already taken care of in other parts of the code, so not necessary for other functions.
    # I just got tired of doing this manually.
    shaft_to_attenuator = 0.875 #mm. since G10 shaft, hence rotation axis, is actually about 1 mm below lower part of "attenuator" part of collimator
    shaft_to_bottom = 7.875 #mm Perpendicular distance between bottom edge of collimator and g10 shaft
    shaft_to_top = 7.775 # mm. Perpendicular distance between top part of attenuator and g10 shaft
    atten_Radius = 31.6/2 # mm.radius of the attenuator part of collimator
    coll_radius = 6.3/2 # mm. radius of lower (collimator) part of collimator


    coll_eff_Radius = np.sqrt(atten_Radius**2+shaft_to_attenuator**2) # in mm. since G10 shaft, hence rotation axis, is actually about 1 mm below lower part of "attenuator" part of collimator, offset by 1 mm, get the hypotenuse for "effective radius"
    shaft_to_topCorner = np.sqrt(atten_Radius**2 + shaft_to_top**2)
    shaft_to_bottomCorner = np.sqrt(coll_radius**2 + shaft_to_bottom**2)

    print('Important clearances! \n Effective collimator radius (distance b/w shaft and side corner): %.2f mm \n Shaft to top corner: %.2f mm \n Shaft to bottom corner: %.2f mm' %(coll_eff_Radius, shaft_to_topCorner, shaft_to_bottomCorner))


if __name__ == '__main__':
	main()
