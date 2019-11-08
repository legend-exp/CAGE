import numpy as np
import matplotlib
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import ROOT
import sys
matplotlib.rcParams['text.usetex'] = True

def main():

	# data_dir = '../alpha/raw_out/'
	# file = 'test_e100000.hdf5'
	# filename = data_dir + file

	# filename = '../alpha/raw_out/uncollimated_test_241Am_700000.hdf5'
	# filename = '../alpha/raw_out/collimated_test_241Am_700000.hdf5'
	# filename = '../alpha/raw_out/30mm_collimated_241Am_700000.hdf5'
	filename = '../alpha/raw_out/30mm_notcollimated_241Am_700000.hdf5'




	#filename = '../alpha/raw_out/test_sebColl_e100000.hdf5'

	plotHist(filename)
	# plotSpot(filename)



	# pandarize(filename)
	# test_func(filename)
	# exit()

	# test3(filename)
	# test1(filename)
	# test2()
	# test4(filename)

# def test4(filename):



def get_hist(np_arr, bins=None, range=None, dx=None, wts=None):
    """
    """
    if dx is not None:
        bins = int((range[1] - range[0]) / dx)

    if bins is None:
        bins = 100 #override np.histogram default of just 10

    hist, bins = np.histogram(np_arr, bins=bins, range=range, weights=wts)
    hist = np.append(hist, 0)

    if wts is None:
        return hist, bins, hist
    else:
        var, bins = np.histogram(np_arr, bins=bins, weights=wts*wts)
        return hist, bins, var

def plotHist(filename):
	df = pandarize(filename)
	energy = np.array(df['energy']*1000)
	x = np.array(df['x'])
	y = np.array(df['y'])
	z = np.array(df['z'])

	fig, ax = plt.subplots()
	plt.hist(energy, range = [0.0, 6000], bins=1000)
	plt.yscale('log')
	# plt.title('Collimated, $^{241}$Am 7*10$^5$ Primaries, 16 mm above detector', fontsize=18)
	plt.title('un-Collimated, $^{241}$Am 7*10$^5$ Primaries, 16 mm above detector', fontsize=18)
	plt.show()

def plotSpot(filename):
	df = pandarize(filename)
	energy = np.array(df['energy']*1000)
	x = np.array(df['x'])
	y = np.array(df['y'])
	z = np.array(df['z'])

	fig, ax = plt.subplots()
	spot_hist = ax.hist2d(x, y, range = [[-32., 32.],[-32., 32.]], norm=LogNorm(), bins=6000) #, range = [[-20., 20.],[-20., 20.]]
	plt.colorbar(spot_hist[3], ax=ax)
	# plt.title('Collimated, $^{241}$Am 7*10$^5$ Primaries, 16 mm above detector', fontsize=18)
	plt.title('un-Collimated, $^{241}$Am 7*10$^5$ Primaries, 16 mm above detector', fontsize=18)
	plt.show()

def test1(filename):

	# energy resolution
	pctResAt1MeV = 0.15 #


	df = pandarize(filename)
	# df.to_hdf('processed_'+ file, key='procdf')
	# exit()
	# print(df.keys())
	# exit()
	energy = np.array(df['energy']*1000)
	x = np.array(df['x'])
	y = np.array(df['y'])
	z = np.array(df['z'])
	# print(x[:])
	# exit()
	# print(len(energy))
	# print(len(x))
	# exit()
	# print(energy[:])
	# exit()
	# print(energy)
	# x = np.array(df['x'])
	# print(x)
	# print('hi')
	# exit()
	# print(energy)
	# print(len(energy))
	fig, ax = plt.subplots()
	# plt.figure()
	# energy = df['energy']*1000
	# energy = df['energy'] + np.sqrt(df['energy'])*pctResAt1MeV/100.*np.random.randn(len(df['energy']))
	# (df['energy']*1000).hist(bins=1000)
	# energy_hist = np.histogram(energy, bins=np.arange(0,5,10))
	# plt.hist(energy, bins=1000)
	# plt.hist(x, bins=100)

	# spot_hist = ax.hist2d(x, y, range = [[-32., 32.],[-32., 32.]], norm=LogNorm(), bins=6000) #, range = [[-20., 20.],[-20., 20.]]
	# plt.colorbar(spot_hist[3], ax=ax)

	plt.hist(energy, range = [0.0, 6000], bins=1000)
	# plt.hist(energy)
	# plt.xlim(0, 6000)
	plt.yscale('log')
	# plt.xscale('log')
	plt.title('Un-collimated, phys vol in place $^{241}$Am, 7*10$^5$ Primaries, 20mm above', fontsize=18)
	# plt.title('Energy Spectrum plane 500 keV e-, 10$^5$ Primaries', fontsize=18)
	# plt.title('Spot from plane 500 keV e-, z = 0 mm, 10$^5$ Primaries', fontsize=18)
	plt.show()

	# print(len(energy))
	# fig = plt.figure()
	# plt.plot(hist)
	# plt.show()
	# plotSpectrum(df)


def pandarize(filename):
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

	procdf= pd.DataFrame(detector_hits.groupby(['event','volID'], as_index=False)['Edep','x_weights','y_weights', 'z_weights'].sum())
    # rename the summed energy depositions for each step within the event to "energy". This is analogous to the event energy you'd see in your detector

	procdf = procdf.rename(columns={'Edep':'energy'})

	procdf['x'] = procdf['x_weights']/procdf['energy']
	procdf['y'] = procdf['y_weights']/procdf['energy']
	procdf['z'] = procdf['z_weights']/procdf['energy']

	del procdf['x_weights']
	del procdf['y_weights']
	del procdf['z_weights']

	# xarr = np.array(procdf['volID'][:])
	# print(xarr)
	# exit()



	# print(procdf['x'])

	return procdf


def test_func(filename):
	g4sfile = h5py.File(filename, 'r')
	g4sntuple = g4sfile['default_ntuples']['g4sntuple']
	col_names = list(g4sntuple.keys())


	g4sdf = pd.DataFrame(np.array(g4sntuple['event']['pages']), columns=['event'])
	for name in col_names:
		col = g4sntuple[name]
		if isinstance(col, h5py.Dataset):
			# might want to handle these differently at some point
			continue
		g4sdf[name] = pd.Series(np.array(col['pages']), index=g4sdf.index)

	# pd.to_hdf(g4sdf, "complvel")
	# print(g4sdf)

	# apply E cut / detID cut and sum Edeps for each event using loc, groupby, and sum
	# write directly into output dataframe
	detector_hits = g4sdf.loc[(g4sdf.Edep > 1e-6)&(g4sdf.volID==1)]#.copy()

	print(detector_hits[['Edep','x','y','z']])

	myarr = detector_hits["Edep"].values

	yv, xv, _ = get_hist(myarr, range=(0, 5), dx=0.0001)


	plt.semilogy(xv, yv, 'r', ls='steps')
	plt.show()

	# procdf = pd.DataFrame(detector_hits.groupby(['event','volID','iRep'], as_index=False)['Edep'].sum()) #gives new df with sum energy

	# procdf['x'] = detector_hits.groupby(['event','volID','iRep'], as_index=False)['x']
	# procdf['y'] = detector_hits.groupby(['event','volID','iRep'], as_index=False)['y']
	# procdf['z'] = detector_hits.groupby(['event','volID','iRep'], as_index=False)['z']
	# procdf = procdf.join(pd.DataFrame(detector_hits.groupby(['event','volID','iRep'], as_index=False)['x']))
	# procdf = procdf.join(pd.DataFrame(detector_hits.groupby(['event','volID','iRep'], as_index=False)['y']))
	# procdf = procdf.join(pd.DataFrame(detector_hits.groupby(['event','volID','iRep'], as_index=False)['z']))
	# procdf = procdf.rename(columns={'iRep':'detID', 'Edep':'energy'})
	# return procdf



if __name__ == '__main__':
	main()
