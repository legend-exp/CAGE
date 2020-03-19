import numpy as np
import matplotlib.pyplot as plt
import sys

# This code allows one to input a source's nndc data written in a .txt file with the columns [particle type] [energy in keV] [intensity], and output a .mac file that can be used to run the source in g4simple.

def main():
 
    if(len(sys.argv) != 3):
        print('Usage: nndc_generate.py [input file name] [output file name]')
        sys.exit()

    spherical_source()

def spherical_source():

    data = [list(map(str,l.split())) for l in open(str(sys.argv[1]))  if not l.startswith('RT')]

    # volume to confine source to, radius of sphere, and where to center the source
    source_confine = 'Source_Disk_PV'
    source_radius = '2.2 mm'
    source_centre = '0 0 56.065 mm'

    with open(str(sys.argv[2]), 'w') as f:
        for i in range(len(data)):
            if i==0:
                f.write('/gps/source/intensity '+str(data[i][2])+'\n/gps/particle '+str(data[i][0])+'\n/gps/ang/type iso \n/gps/ene/mono '+str(data[i][1])+' keV \n/gps/pos/type Volume \n/gps/pos/shape Sphere \n/gps/pos/radius '+str(source_radius)+'\n/gps/pos/centre '+str(source_centre)+'\n/gps/pos/confine '+str(source_confine)+'\n\n')
            else:
                f.write('/gps/source/add '+str(data[i][2])+'\n/gps/particle '+str(data[i][0])+'\n/gps/ang/type iso \n/gps/ene/mono '+str(data[i][1])+' keV \n/gps/pos/type Volume \n/gps/pos/shape Sphere \n/gps/pos/radius '+str(source_radius)+'\n/gps/pos/centre '+str(source_centre)+'\n/gps/pos/confine '+str(source_confine)+'\n\n')

if __name__ == '__main__':
        main()


