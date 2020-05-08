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

def main():

    filename = '../alpha/raw_out/newDet_sourceRotNorm_y6mm_ICPC_Pb_241Am_20000000.hdf5'
    processed_filename = '../alpha/processed_out/processed_newDet_sourceRotNorm_y6mm_ICPC_Pb_241Am_20000000.hdf5'

    post_process(filename, processed_filename, source=False)


def post_process(filename, processed_filename, source=False):
	print('Processing file: ', filename)
	if source==True:
		procdf, sourcePV_df = pandarize(filename, source)
		# df.to_hdf('../alpha/processed_out/processed_newDet_test.hdf5', key='procdf', mode='w')
		procdf.to_hdf(processed_filename, key='procdf', mode='w')
		sourcePV_df.to_hdf(processed_filename, key='sourcePV_df', mode='w')
	else:
		procdf = pandarize(filename, source)
		# df.to_hdf('../alpha/processed_out/processed_newDet_test.hdf5', key='procdf', mode='w')
		procdf.to_hdf(processed_filename, key='procdf', mode='w')

	print('File processed. Output saved to: ', processed_filename)

def pandarize(filename, source=False):
	# have to open the input file with h5py (g4 doesn't write pandas-ready hdf5)
	g4sfile = h5py.File(filename, 'r')
	g4sntuple = g4sfile['default_ntuples']['g4sntuple']

	# build a pandas DataFrame from the hdf5 datasets we will use
	# list(g4sfile['default_ntuples']['g4sntuple'].keys())=>['Edep','KE','columns','entries','event',
	# 'forms','iRep','lx','ly','lz','nEvents','names','parentID','pid','step','t','trackID','volID','x','y','z']

	g4sdf = pd.DataFrame(np.array(g4sntuple['event']['pages']), columns=['event'])
	g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['step']['pages']), columns=['step']),
	                   lsuffix = '_caller', rsuffix = '_other')
	g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['Edep']['pages']), columns=['Edep']),
	                   lsuffix = '_caller', rsuffix = '_other')
	g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['volID']['pages']),
	                   columns=['volID']), lsuffix = '_caller', rsuffix = '_other')
	g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['iRep']['pages']),
	                   columns=['iRep']), lsuffix = '_caller', rsuffix = '_other')
	g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['pid']['pages']),
                       columns=['pid']), lsuffix = '_caller', rsuffix = '_other')
	g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['x']['pages']),
                       columns=['x']), lsuffix = '_caller', rsuffix = '_other')
	g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['y']['pages']),
                       columns=['y']), lsuffix = '_caller', rsuffix = '_other')
	g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['z']['pages']),
                       columns=['z']), lsuffix = '_caller', rsuffix = '_other')


	# print(g4sdf)
	# xarr = np.array(g4sdf['volID'])
	# print(xarr[:])
	# print(type(g4sntuple['x']['pages']))
	# exit()

	# apply E cut / detID cut and sum Edeps for each event using loc, groupby, and sum
	# write directly into output dataframe
	detector_hits = g4sdf.loc[(g4sdf.Edep>0)&(g4sdf.volID==1)]

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


	#Do same as above with PV that defines where the source should be if set source PV in macro

	if source==True:
		source_hits = g4sdf.loc[(g4sdf.Edep>0)&(g4sdf.volID==2)]

		source_hits['x_weights'] = source_hits['x'] * source_hits['Edep']
		source_hits['y_weights'] = source_hits['y'] * source_hits['Edep']
		source_hits['z_weights'] = source_hits['z'] * source_hits['Edep']

		sourcePV_df= pd.DataFrame(source_hits.groupby(['event','volID'], as_index=False)['Edep','x_weights','y_weights', 'z_weights', 'pid'].sum())
		sourcePV_df = sourcePV_df.rename(columns={'Edep':'energy'})

		sourcePV_df['x'] = sourcePV_df['x_weights']/sourcePV_df['energy']
		sourcePV_df['y'] = sourcePV_df['y_weights']/sourcePV_df['energy']
		sourcePV_df['z'] = sourcePV_df['z_weights']/sourcePV_df['energy']

		del sourcePV_df['x_weights']
		del sourcePV_df['y_weights']
		del sourcePV_df['z_weights']

		return procdf, sourcePV_df

	else:
		return procdf


if __name__ == '__main__':
	main()
