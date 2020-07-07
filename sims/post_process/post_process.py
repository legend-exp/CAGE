import numpy as np
import scipy
import matplotlib
from matplotlib.colors import LogNorm
from scipy.stats import norm, kde
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import h5py
import pandas as pd
#import ROOT
import sys
#from particle import PDGID
#matplotlib.rcParams['text.usetex'] = True

def main():

    # filename = '../alpha/raw_out/newDet_sourceRot25_thetaDet65_y14mm_ICPC_Pb_241Am_100000000.hdf5'
    # processed_filename = '../alpha/processed_out/processed_newDet_sourceRot25_thetaDet65_y14mm_ICPC_Pb_241Am_100000000.hdf5'

    # raw_dir = '../alpha/raw_out/'
    # processed_dir = '../alpha/processed_out/'
    # base_filenames = ['test_newDet_sourceRotNorm_y6mm_ICPC_Pb_241Am_100000.hdf5']

    # post_process(filename, processed_filename)

    raw_dir = '../alpha/raw_out/oppi/'
    processed_dir = '../alpha/processed_out/oppi/'
    base_filenames = ['test_oppi_y19mm_norm_241Am_1000000.hdf5']
    # base_filenames =['oppi_y19mm_norm_241Am_100000000.hdf5', 'oppi_y22mm_norm_241Am_100000000.hdf5', 'oppi_y25mm_norm_241Am_100000000.hdf5', 'oppi_y28mm_norm_241Am_100000000.hdf5', 'oppi_y31mm_norm_241Am_100000000.hdf5']


    for file in range(len(base_filenames)):
        post_process(raw_dir, processed_dir, base_filenames[file])


def post_process(raw_dir, processed_dir, base_filename, hits=False):
    filename = raw_dir+base_filename
    processed_filename = processed_dir+'processed_'+base_filename
    print('Processing file: ', filename)
    # print(processed_filename)
    # exit()
    if hits==True:
        procdf, pos_df = pandarize(filename, hits=True)
        # df.to_hdf('../alpha/processed_out/processed_newDet_test.hdf5', key='procdf', mode='w')
        procdf.to_hdf(processed_filename, key='procdf', mode='w')
        pos_df.to_hdf(processed_filename, key='pos_df', mode='w')
    else:
        procdf = pandarize(filename, hits=False)
        # df.to_hdf('../alpha/processed_out/processed_newDet_test.hdf5', key='procdf', mode='w')
        procdf.to_hdf(processed_filename, key='procdf', mode='w')

    print('File processed. Output saved to: ', processed_filename)


def pandarize(filename, hits=False):
    # have to open the input file with h5py (g4 doesn't write pandas-ready hdf5)
    g4sfile = h5py.File(filename, 'r')
    g4sntuple = g4sfile['default_ntuples']['g4sntuple']

    # build a pandas DataFrame from the hdf5 datasets we will use
    # list(g4sfile['default_ntuples']['g4sntuple'].keys())=>['Edep','KE','columns','entries','event',
    # 'forms','iRep','lx','ly','lz','nEvents','names','parentID','pid','step','t','trackID','volID','x','y','z']

    g4sdf = pd.DataFrame(np.array(g4sntuple['event']['pages']), columns=['event'])

    g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['step']['pages']), columns=['step']), lsuffix = '_caller', rsuffix = '_other')
    g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['Edep']['pages']), columns=['Edep']), lsuffix = '_caller', rsuffix = '_other')
    g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['volID']['pages']), columns=['volID']), lsuffix = '_caller', rsuffix = '_other')
    g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['iRep']['pages']), columns=['iRep']), lsuffix = '_caller', rsuffix = '_other')
    g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['pid']['pages']), columns=['pid']), lsuffix = '_caller', rsuffix = '_other')
    g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['x']['pages']), columns=['x']), lsuffix = '_caller', rsuffix = '_other')
    g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['y']['pages']), columns=['y']), lsuffix = '_caller', rsuffix = '_other')
    g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['z']['pages']), columns=['z']), lsuffix = '_caller', rsuffix = '_other')

    # detector_hits = g4sdf.loc[(g4sdf.Edep>1.e-6)&(g4sdf.volID==1)]
    detector_hits = g4sdf.loc[(g4sdf.volID==1)]

    detector_hits['x_weights'] = detector_hits['x'] * detector_hits['Edep']
    detector_hits['y_weights'] = detector_hits['y'] * detector_hits['Edep']
    detector_hits['z_weights'] = detector_hits['z'] * detector_hits['Edep']

    procdf= pd.DataFrame(detector_hits.groupby(['event','volID'], as_index=False)['Edep','x_weights','y_weights', 'z_weights', 'pid'].sum())

    # rename the summed energy depositions for each step within the event to "energy". This is analogous to the event energy you'd see in your detector
    procdf = procdf.rename(columns={'Edep':'energy'})

    procdf['x'] = procdf['x_weights']/procdf['energy']
    procdf['y'] = procdf['y_weights']/procdf['energy']
    procdf['z'] = procdf['z_weights']/procdf['energy']

    del procdf['x_weights']
    del procdf['y_weights']
    del procdf['z_weights']

    return procdf


    if hits==True:
        # apply E cut / detID cut and sum Edeps for each event using loc, groupby, and sum
    	# write directly into output dataframe
        detector_hits = g4sdf.loc[(g4sdf.Edep>1.e-6)&(g4sdf.volID==1)]
        eventArr = []
        for eventNum in detector_hits['event']:
            if eventNum not in eventArr:
                eventArr.append(eventNum)
        energies = []
        r_arr = []
        phi_arr = []
        z_arr= []

        for eventNum in eventArr:
            temp_df = detector_hits.loc[(detector_hits.event==eventNum)]
            energies.append(np.array(temp_df['Edep']))
            x = (np.array(temp_df['x']))
            y = (np.array(temp_df['y']))
            z = (np.array(temp_df['z']))
            r = np.sqrt(x**2+y**2)
            phi = np.arctan(y/x)
            r_arr.append(r)
            phi_arr.append(phi)
            z_arr.append(z)

        energies = np.array(energies)
        phi_arr = np.array(phi_arr)
        r_arr = np.array(r_arr)
        z_arr = np.array(z_arr)

        new_eventNum_arr = np.arange(len(eventArr))
        pos_df = pd.DataFrame(new_eventNum_arr, columns=['event'])
        pos_df = pos_df.join(pd.DataFrame(energies, columns=['energy']))
        pos_df = pos_df.join(pd.DataFrame(r_arr, columns=['r']))
        pos_df = pos_df.join(pd.DataFrame(phi_arr, columns=['phi']))
        pos_df = pos_df.join(pd.DataFrame(z_arr, columns=['z']))

        return pos_df

        # print(g4sdf)
        # xarr = np.array(g4sdf['volID'])
        # print(xarr[:])
        # print(type(g4sntuple['x']['pages']))
        # exit()


if __name__ == '__main__':
	main()
