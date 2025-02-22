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
    radius = [14]
    source_angle = [45, 60, 75, 90]
    rotary = [0, 180, 145]
    mac_dir = './macros/'
    gdml_dir = './geometries/mothers/'
    hdf5_dir = '$[TMPDIR]/' #format environment variables as $[VARNAME] 
    run = 'ditch_scan/' #ex 'centering_scan/'
    det = 'icpc'
    primaries = 1000000
    jobs = 100
    writeFiles(radius, source_angle, rotary, det, run, primaries, jobs, hdf5_dir=hdf5_dir, write_shell=True, run_job=False)


def writeFiles(radius, source_angle, rotary='0', det='oppi', run = '', primaries=100000000, jobs=10, mac_dir = './macros/', gdml_dir = './geometries/mothers/', hdf5_dir = './alpha/raw_out/', write_shell=False, run_job=False):
    batch_job = []
    for r in radius:
        for theta_det in source_angle:
            for theta_rot in rotary:
                print(f'prepping sims files for {det}: r: {r}, theta_det {theta_det}, theta_rot: {theta_rot}')
                gdml_out_file = gdml_dir + f'{det}/{run}y{int(r)}_thetaDet{int(theta_det)}_rotary{int(theta_rot)}_241Am_{primaries}.gdml'
                mac_out_file = mac_dir + f'{det}/{run}y{int(r)}_thetaDet{int(theta_det)}_rotary{int(theta_rot)}_241Am_{primaries}.mac'
                hdf5_out_file = hdf5_dir+ f'y{int(r)}_thetaDet{int(theta_det)}_rotary{int(theta_rot)}_241Am_{primaries}.hdf5'

                #Create directory for hdf5 output if doesn't alreasy exist
                #if not os.path.isdir(hdf5_dir):
                #    print(f'Creating directory for hdf5 output file: {hdf5_dir}')
                #    os.mkdir(hdf5_dir)
                    
                with open(gdml_dir + f'{det}/template.gdml', 'r') as file:
                    # read a list of lines into data
                    gdml = file.readlines()

                with open(mac_dir + f'{det}/template.mac', 'r') as file:
                    # read a list of lines into data
                    mac = file.readlines()

                if det=='icpc':
                    icpc = True
                else: 
                    icpc = False
                mac_rotation, gdml_source_center, gdml_source_rotation = positionCalc(r, theta_det, icpc)
                gdml_divingBoard_rotation = f'     <rotation name="ICPC_diving_board_volume_Rotation" z="-(90+17.92)-{theta_rot}" unit="deg"/> <!-- diving board rotation. linear motor drive --> \n'
                gdml_det_rotation = f'     <rotation name="ICPC_Rotation" x="0" y="0" z="-{theta_rot}" unit="deg"/> <!--Add same additional rotation to "OPPI1_diving_board_volume_Rotation" here to simulte rotating rotary motor--> \n'

                gdml[84] = gdml_source_center
                gdml[85] = gdml_source_rotation
                gdml[97] = gdml_divingBoard_rotation
                gdml[103] = gdml_det_rotation

                mac[9] = '/g4simple/setDetectorGDML ' + gdml_out_file + ' 0 \n'
                mac[13] = '/g4simple/setFileName ' + hdf5_out_file + ' \n'

                if int(theta_det)==90:
                    mac[43] = f'/gps/pos/centre {float(r)} 0.0 3.85 mm \n'

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
                    with open(f'./submission/run_template.sh', 'r') as file:
                        # read a list of lines into data
                        runsh = file.readlines()
                    with open(f'./submission/submit_template.sh', 'r') as file:
                        # read a list of lines into data
                        sub = file.readlines()
                    run_name = f'run_y{int(r)}_thetaDet{int(theta_det)}_rot{int(theta_rot)}.sh'
                    sub_name = f'sub_y{int(r)}_thetaDet{int(theta_det)}_rot{int(theta_rot)}.sh'

                    runsh[7] = f'g4simple {mac_out_file} \n'
                    runsh[9] = f'h5repack -v -f GZIP=5 y{int(r)}_thetaDet{int(theta_det)}_rotary{int(theta_rot)}_241Am_{primaries}.hdf5 ' + '${DATADIR}/' + f'y{int(r)}_thetaDet{int(theta_det)}_rotary{int(theta_rot)}_' + '${SGE_TASK_ID}.hdf5 \n'

                    sub[6] = f'#$ -N y{int(r)}_thetaDet{int(theta_det)}_rotary{int(theta_rot)} #job name \n'
                    sub[12] = f'#$ -t 1-{jobs} #give me N identical jobs, labelled by variable SGE_TASK_ID \n'
                    sub[16] = f'singularity exec --bind /data/eliza1/LEGEND,$TMPDIR /data/eliza1/LEGEND/sw/containers/legend-base.sif /data/eliza1/LEGEND/users/grsong/CAGE/new_sims/submission/{run_name} \n'

                    with open(f'./submission/{run_name}', 'w') as file:
                        # read a list of lines into data
                        file.writelines(runsh)
                    result = subprocess.run([f'chmod a+x ./submission/{run_name}'], shell=True, check=True, capture_output=True, text=True).stdout.strip("\n")
                    print(result)
                    with open(f'./submission/{sub_name}', 'w') as file:
                        # read a list of lines into data
                        file.writelines(sub)



if __name__ == '__main__':
	main()
