import numpy as np
import h5py
import pandas as pd
import sys
import matplotlib.pyplot as plt
plt.style.use('style.mplstyle')

def main():

    #spectrum_with_dead_layer()
    spectrum()
    #spectrum_for_muons_in_veto()
    #spectrum_for_muons_with_veto()

def spectrum_with_dead_layer():
    
    """
    Run postprocesshdf5.py on g4simple output file to get desired files for the dataframes defined below.
    """ 

    if(len(sys.argv) != 3):
        print('Usage: spectrum.py [maximum energy value for x axis of plot in keV] [Source]')
        sys.exit()

    df1 = pd.read_hdf("processed_with_dead_layer.hdf5", key="procdf")
    df2 = pd.read_hdf("processed.hdf5", key="procdf")

    m = list(df1['energy'])
    p = list(x*1000 for x in m)

    n = list(df2['energy'])
    q = list(x*1000 for x in n)

    plt.hist(p, np.arange(0,int(sys.argv[1]),0.2), histtype='step', color = 'white', label='{} entries'.format(len(p)))
    plt.hist(p, np.arange(0,int(sys.argv[1]),0.2), histtype='stepfilled', color = 'aqua', label='with dead layer')
    plt.hist(q, np.arange(0,int(sys.argv[1]),0.2), histtype='step', color = 'black', label='without dead layer')
    plt.xlim(0,int(sys.argv[1]))
    #plt.ylim(0,plt.ylim()[1])    
    plt.xlabel('Energy (keV)', ha='right', x=1.0)
    plt.ylabel('Counts', ha='right', y=1.0)
    plt.title('Energy Spectrum ('+sys.argv[2]+' Source on Passivated Surface)')
    plt.legend(frameon=True, loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.semilogy()
    plt.show()

def spectrum():

    """
    Run postprocesshdf5.py on g4simple output file to get desired files for the dataframes defined below.
    """ 

    if(len(sys.argv) != 3):
        print('Usage: spectrum.py [maximum energy value for x axis of plot in keV] [Source]')
        sys.exit()

    df =  pd.read_hdf("processed.hdf5", key="procdf")

    m = list(df['energy'])
    p = list(x*1000 for x in m)

    plt.hist(p, np.arange(0,int(sys.argv[1]),0.1), histtype='step', color = 'black', label='{} entries'.format(len(p)))
    plt.xlim(0,int(sys.argv[1]))
    #plt.ylim(0,plt.ylim()[1])
    plt.xlabel('Energy (keV)', ha='right', x=1.0)
    plt.ylabel('Counts', ha='right', y=1.0)
    plt.title('Energy Spectrum ('+sys.argv[2]+' Source + LANL Geometry w/ 10 micron Au foil)')
    plt.legend(frameon=True, loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.semilogy()
    #plt.semilogx()
    plt.show()

def spectrum_for_muons_in_veto():

    """
    Run postprocesshdf5.py on g4simple output file to get desired files for the dataframes defined below.
    """ 
    if(len(sys.argv) != 2):
        print('Usage: spectrum.py [veto number]')
        sys.exit()

    df =  pd.read_hdf("muon_processed_in_veto_"+sys.argv[1]+".hdf5", key="procdf")

    m = list(df['energy'])

    plt.hist(m, np.arange(0,300,0.01), histtype='step', color = 'black', label='veto '+str(sys.argv[1])+', {} entries'.format(len(m)))
    plt.xlim(0,275)
    #plt.ylim(0,plt.ylim()[1])
    plt.xlabel('Energy (MeV)', ha='right', x=1.0)
    plt.ylabel('Counts', ha='right', y=1.0)
    plt.title('Energy Spectrum Measured by Scintillator (Muon Distribution)')
    plt.legend(frameon=True, loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.semilogy()
    plt.show()

def spectrum_for_muons_with_veto():

    """
    Run postprocesshdf5.py on g4simple output file to get desired files for the dataframes defined below.
    """

    if(len(sys.argv) != 2):
        print('Usage: spectrum.py [veto number]')
        sys.exit()

    df1 =  pd.read_hdf("muon_processed_without_veto_"+sys.argv[1]+".hdf5", key="procdf")
    df2 =  pd.read_hdf("muon_processed_with_veto_"+sys.argv[1]+".hdf5", key="procdf")

    m1 = list(df1['energy'])
    p1 = list(x*1000 for x in m1)
    m2 = list(df2['energy'])
    p2 = list(x*1000 for x in m2)

    plt.hist(p1, np.arange(5,4001,1), histtype='step', color = 'black', label='no veto, {} entries'.format(len(p1)))
    plt.hist(p2, np.arange(5,4001,1), histtype='step', color = 'aqua', label='veto with 3 MeV threshold, {} entries'.format(len(p2)))
    plt.xlim(0,4000)
    #plt.ylim(0,plt.ylim()[1])
    plt.xlabel('Energy (keV)', ha='right', x=1.0)
    plt.ylabel('Counts', ha='right', y=1.0)
    plt.title('Energy Spectrum (Muon Distribution)')
    plt.legend(frameon=True, loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.semilogy()
    plt.show()

if __name__ == '__main__':
	main()

