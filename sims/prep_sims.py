import numpy as np
import scipy
import matplotlib as mpl
import sys
import os
import math
import subprocess
mpl.rcParams['text.usetex'] = True
mpl.use('Agg')

def main():
    doc="""

    Write GDML mother geometry, g4simple macro, and job submission commands for sims

    G. Othman
    """
    radius = [12]
    source_angle = [90.0]
    rotary = np.linspace(4, 144, 15)
    # print(rotary)
    # exit()
    mac_dir = './macros/'
    gdml_dir = './geometries/mothers/'
    hdf5_dir = './alpha/raw_out/'
    run = 'rotary_centering_scan/' #ex 'centering_scan/'
    det = 'oppi'
    primaries = 100000000
    # print(f'./geometries/mothers/{det}/{run}test.gdml')
    writeFiles(radius, source_angle, rotary, det, run, primaries, write_shell=True, run_job=False)


def writeFiles(radius, source_angle, rotary='0', det='oppi', run = '', primaries=100000000, mac_dir = './macros/', gdml_dir = './geometries/mothers/', hdf5_dir = './alpha/raw_out/', write_shell=False, run_job=False):
    batch_job = []
    for r in radius:
        for theta_det in source_angle:
            for theta_rot in rotary:
                print(f'prepping sims files for {det}: r: {r}, theta_det {theta_det}, theta_rot: {theta_rot}')
                gdml_out_file = gdml_dir + f'{det}/{run}y{r}_thetaDet{int(theta_det)}_rotary{int(theta_rot)}_241Am_{primaries}.gdml'
                mac_out_file = mac_dir + f'{det}/{run}y{r}_thetaDet{int(theta_det)}_rotary{int(theta_rot)}_241Am_{primaries}.mac'
                hdf5_out_file = hdf5_dir+ f'{det}/{run}y{r}_thetaDet{int(theta_det)}_rotary{int(theta_rot)}_241Am_{primaries}.hdf5'

                #Create directory for hdf5 output if doesn't alreasy exist
                if not os.path.isdir(hdf5_dir +f'{det}/{run}'):
                    print(f'Creating directory for hdf5 output file: {gdml_dir + det +"/"+run}')
                    os.mkdir(hdf5_dir +f'{det}/{run}')


                with open(gdml_dir + f'{det}/template.gdml', 'r') as file:
                    # read a list of lines into data
                    gdml = file.readlines()

                with open(mac_dir + f'{det}/template.mac', 'r') as file:
                    # read a list of lines into data
                    mac = file.readlines()

                mac_rotation, gdml_source_center, gdml_source_rotation = positionCalc(r, theta_det)
                gdml_divingBoard_rotation = f'     <rotation name="OPPI1_diving_board_volume_Rotation" z="(180-17.92)-{theta_rot}" unit="deg"/> <!-- diving board rotation. linear motor drive \n'
                gdml_det_rotation = f'     <rotation name="OPPI_Rotation" x="0" y="0" z="-{theta_rot}" unit="deg"/> <!--Add same additional rotation to "OPPI1_diving_board_volume_Rotation" here to simulte rotating rotary motor--> \n'

                gdml[89] = gdml_source_center
                gdml[90] = gdml_source_rotation
                gdml[102] = gdml_divingBoard_rotation
                gdml[109] = gdml_det_rotation

                mac[9] = '/g4simple/setDetectorGDML ' + gdml_out_file + ' \n'
                mac[13] = '/analysis/setFileName ' + hdf5_out_file + ' \n'

                if int(theta_det)==90:
                    mac[43] = f'/gps/pos/centre 0.0 {float(r)} 4.5 mm \n'

                else:
                    mac[43] = ''
                    mac[46] = mac_rotation

                mac[49] = f'/run/beamOn {primaries}'


                if os.path.isdir(gdml_dir +f'{det}/{run}'):
                    with open(gdml_out_file, 'w') as file:
                        # read a list of lines into data
                        file.writelines(gdml)
                else:
                    os.mkdir(gdml_dir +f'{det}/{run}')
                    print(f'Creating directory for {gdml_dir + det +"/"+run} before writing!')
                    with open(gdml_out_file, 'w') as file:
                        # read a list of lines into data
                        file.writelines(gdml)

                if os.path.isdir(mac_dir + f'{det}/{run}'):
                    with open(mac_out_file, 'w') as file:
                        # read a list of lines into data
                        file.writelines(mac)
                else:
                    os.mkdir(mac_dir + f'{det}/{run}')
                    print(f'Creating directory for {mac_dir + det +"/"+run} before writing!')
                    with open(mac_out_file, 'w') as file:
                        # read a list of lines into data
                        file.writelines(mac)

                if write_shell:
                    # cori_command = f'echo $SHELL \n' #just do this for testing
                    cori_command = f'sbatch --chdir=/global/homes/g/gothman/projecta/joule_CAGE/sims/ --output=/global/homes/g/gothman/projecta/joule_CAGE/sims/{det}_y{r}_thetaDet{int(theta_det)}_rotary{int(theta_rot)}_241Am_cori-%j.txt --image=legendexp/legend-software:latest -C haswell --export=HDF5_USE_FILE_LOCKING=FALSE --qos=shared -t 48:00:00 slurm.slr "g4simple {mac_out_file}" \n'
                    batch_job.append(cori_command)

    if write_shell:
        with open('./run_sim_jobs.sh', 'w') as file:
            file.writelines(batch_job)

    if run_job:
        result1 = subprocess.run(["chmod +x sim_jobs.sh"], shell=True, check=True, capture_output=True, text=True).stdout.strip("\n")
        result = subprocess.run(["./run_sim_jobs.sh"], shell=True, check=True, capture_output=True, text=True).stdout.strip("\n")
        print(result)



def writeShell():

    command = f'sbatch --chdir=/global/homes/g/gothman/projecta/joule_CAGE/sims/ --output=/global/homes/g/gothman/projecta/joule_CAGE/sims/{det}_y{r}_thetaDet{int(theta_det)}_rotary{int(theta_rot)}_241Am_cori-%j.txt --image=legendexp/legend-software:latest -C haswell --export=HDF5_USE_FILE_LOCKING=FALSE --qos=shared -t 48:00:00 slurm.slr "g4simple {mac_out_file}"'




def positionCalc(y_final, theta_det, det='oppi'):
    theta_rot = 90.-theta_det #rotation angle of the collimator with respect to the horizontal. Used in calculations, where 0 deg theta_rot is normal incidence on the detector surface
    theta_rot_motor = theta_rot - 180. #rotation angle of the collimator with respect to the horizontal. Should be the real-life rotation angle of the source motor, where -180 deg is normal incidence on the detector surface
    pi = math.pi
    deg_to_rad = pi/180.

    # First check that it's safe to rotate to this angle
    rotCheck = checkRotation(theta_det)
    # if rotCheck[0]==False:
        # exit()

    ditch_depth = 2. # ditch depth for ICPC in mm
    rotAxis_toSource_height = 4.5 # height difference in mm from the rotation axis to where the activity is located
    if det=='oppi':
        rotAxis_height = 22.0 # height for OPPI in mm from top of detector to rotation axis, which is (0, 0, 0) in the mother geometry of the simulation
        print('Using OPPI axis height: % .1f' %rotAxis_height)
    elif det=='icpc':
        rotAxis_height = 22.5 # height in mm from top of detector to rotation axis, which is (0, 0, 0) in the mother geometry of the simulation
        print('Using ICPC axis height: % .1f' %rotAxis_height)

    else:
        print('Need to specify a detector!')
        exit()

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

    if det=='icpc':
        if (13.<y_final<16.):
            print('Input final y position (radius) is located within the ditch.\nOffsetting by ditch depth value: %.1f mm \n'   %ditch_depth)
            delta_y_ditch = ditch_depth/(math.tan((theta_det*deg_to_rad)))
        else:
            delta_y_ditch = 0.0

    elif det=='oppi':
        delta_y_ditch = 0.0

    else:
        print('Need to specify a detector!')
        exit()

    # print('delta_y_ditch: ', delta_y_ditch)

    delta_y_tot = delta_y_rotation + delta_y_ditch # total y displacement, wrt the rotation axis, from rotating the source and from final desired y position (y_final) being in the ditch if it is
    axis_yPos = y_final - delta_y_tot # final y-position the axis of the collimator (y-coord of center of "sourceRotationVolume")
    lab_axis_yPos = axis_yPos
    source_yPos = axis_yPos - np.abs(delta_y_source) # final y-position of the source activity within the collimator (specified in g4simple run macro)

    rotUnitVec_y = math.cos(theta_rot*deg_to_rad) # determines the y-coordinate of the rotation vector (/gps/pos/rot2) for rotating the source activity in the g4simple run macro
    rotUnitVec_z = math.sin(theta_rot*deg_to_rad) # determines the z-coordinate of the rotation vector (/gps/pos/rot2) for rotating the source activity in the g4simple run macro

    if axis_yPos < 0.:
        rotary_motor_theta = -180.
        theta_rot_motor = -1*(180.+theta_rot)
        lab_axis_yPos *= -1 # this needs to be positive, since will be driving the linear stage "forward" at -180 deg, but its equivalent to a negative axis_yPos

    else:
        rotary_motor_theta = 0.

    macro_rotation = f'/gps/pos/rot1 1 0 0 \n/gps/pos/rot2 0 {rotUnitVec_y:.5f} {rotUnitVec_z:.5f} \n/gps/pos/centre 0.0 {source_yPos:.3f} {source_zPos:.3f} mm \n'
    gdml_source_center = f'     <position name= "source_center" x="0.0" y="{axis_yPos:.3f}" z="0.0" unit="mm"/>\n'
    gdml_source_rotation = f'     <rotation name="source{theta_rot:.0f}" x="-{theta_rot:.2f}" unit="deg"/>\n'

    if theta_rot ==0.:
        gdml_source_rotation = f'     <rotation name="identity" x="0.0" unit="deg"/>\n'

    print('For theta_det= %.1f deg, radius= %.1f mm:' %(theta_det, y_final))
    print('Source activity in the run macro should be rotated to and centered according to: \n/gps/pos/rot1 1 0 0 \n/gps/pos/rot2 0 %.5f %.5f' %(rotUnitVec_y, rotUnitVec_z))
    print('/gps/pos/centre 0.0 %.3f %.3f mm \n' %( source_yPos, source_zPos))


    print('Position of the source ("sourceRotationVolume") in the mother GDML file, should be placed at: \n<position name= "source_center" x="0.0" y="%.3f" z="0.0" unit="mm"/> \n<rotation name="source%.0f" x="-%.2f" unit="deg"/> \n' %(axis_yPos, theta_rot, theta_rot))

    # print('In the lab, to correspond to theta_det= %.1f deg, at radius= %.1f mm: \nsource motor should be rotated to %.1f deg \nsource should be translated to %.3f mm from center' %(theta_det, y_final, theta_rot_motor, axis_yPos))
    print(f'In the lab, to correspond to theta_det= {theta_det:.1f} deg, at radius= {y_final:.1f} mm: \nrotary motor should be rotated to {rotary_motor_theta:.1f} deg \nsource should be translated to {lab_axis_yPos:.3f} mm from center \nsource motor should be rotated to {theta_rot_motor:.1f} deg ')

    return macro_rotation, gdml_source_center, gdml_source_rotation

def checkRotation(theta_det, min_clearance_toLMFE=5.0, det='oppi'):
    # First check that the source can be rotated to this angle while still maintaining desired LMFE clearance
    # Check maximum rotation angle of the collimator in order to leave "min_clearance_toLMFE" mm clearance between the collimator and the LMFE, with the collimator rotation axis "rotAxis_height" mm above the detector surface, and the LMFE "height_det_to_LMFE" mm above the detector surface.

    rad_to_deg = 180./math.pi
    deg_to_rad = math.pi/180.
    theta_rot = (90 - theta_det)

    if det=='icpc':
        rotAxis_height = 22.5 # height in mm from top of detector to rotation axis, which is (0, 0, 0) in the mother geometry of the simulation
        height_det_to_LMFE = 7.0 # height in mm between hieghest point of LMFE and detector surface
        print('Calculating maximum ratoation angle for ICPC')
    elif det=='oppi':
        rotAxis_height = 22.0 # height in mm from top of detector to rotation axis, which is (0, 0, 0) in the mother geometry of the simulation
        height_det_to_LMFE = 6.0 # height in mm between hieghest point of LMFE and detector surface
        print('Calculating maximum rotation angle for OPPI')
    else:
        print('Need to specify a detector!')
        exit()

    height_LMFE_to_ax = rotAxis_height - height_det_to_LMFE # height in mm between top of LMFE and rotation axis
    #min_clearance_toLMFE = 5. # minimum height in mm to maintain of collimator above LMFE

    coll_Radius = 16 # mm
    coll_eff_Radius = np.sqrt(coll_Radius**2+1.0**2) # in mm. since G10 shaft, hence rotation axis, is actually about 1 mm below lower part of "attenuator" part of collimator, offset by 1 mm, get the hypotenuse for "effective radius"
    delta_z = coll_eff_Radius*math.sin(theta_rot*deg_to_rad) # z-distance in mm the bottom edge of the (attenuator part of the) collimator moves downward due to rotation
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
