import numpy as np
import scipy
import matplotlib as mpl
import sys
import os
import math
import subprocess

from source_placement import *

mpl.rcParams['text.usetex'] = True
mpl.use('Agg')

def main():
    doc="""

    Write GDML mother geometry, g4simple macro, and job submission commands for sims

    G. Othman
    """
    radius = [15.]
    source_angle = [90.]
    rotary = [0.]
#     rotary = np.linspace(4, 144, 15)
    # print(rotary)
    # exit()
    mac_dir = './macros/'
    gdml_dir = './geometries/mothers/'
    hdf5_dir = '$TMPDIR/'
    run = 'centering_scan/' #ex 'centering_scan/'
    det = 'oppi'
    primaries = 100
    # print(f'./geometries/mothers/{det}/{run}test.gdml')
    writeFiles(radius, source_angle, rotary, det, run, primaries, hdf5_dir=hdf5_dir, write_shell=False, run_job=False)


def writeFiles(radius, source_angle, rotary='0', det='oppi', run = '', primaries=100000000, mac_dir = './macros/', gdml_dir = './geometries/mothers/', hdf5_dir = './alpha/raw_out/', write_shell=False, run_job=False):
    batch_job = []
    for r in radius:
        for theta_det in source_angle:
            for theta_rot in rotary:
                print(f'prepping sims files for {det}: r: {r}, theta_det {theta_det}, theta_rot: {theta_rot}')
                gdml_out_file = gdml_dir + f'{det}/{run}y{int(r)}_thetaDet{int(theta_det)}_rotary{int(theta_rot)}_241Am_{primaries}.gdml'
                mac_out_file = mac_dir + f'{det}/{run}y{int(r)}_thetaDet{int(theta_det)}_rotary{int(theta_rot)}_241Am_{primaries}.mac'
                hdf5_out_file = hdf5_dir+ f'y{int(r)}_thetaDet{int(theta_det)}_rotary{int(theta_rot)}_241Am_{primaries}.hdf5'

                #Create directory for hdf5 output if doesn't alreasy exist
                #if not os.path.isdir(hdf5_dir +f'{det}/{run}'):
                #    print(f'Creating directory for hdf5 output file: {gdml_dir + det +"/"+run}')
                #    os.mkdir(hdf5_dir +f'{det}/{run}')
                    
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


if __name__ == '__main__':
	main()
