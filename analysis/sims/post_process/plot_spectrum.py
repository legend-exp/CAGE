import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import ROOT

def main():

	filename = './alpha_5MeV_10000000.hdf5'
	# pandarize(filename)
	test_func(filename)
	exit()

	# test3(filename)
	# test1()
	# test2()


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



def test1():

	# filename = './test.hdf5'
	filename = './alpha_5MeV_10000000.hdf5'
	# filename = '/Users/gothman/analysis/g4simple/Example/g4simpleout.hdf5'

	df = pandarize(filename)
	print(df.keys())

	# energy = np.array(df['energy'])
	x = np.array(df['x'])
	print(x)
	# print(energy)
	# print(len(energy))
	# plt.figure()
	# (df['energy']*1000).hist(bins=100)
	# # hist = np.histogram(energy, bins=np.arange(0,5,10))
	# # plt.hist(energy)
	# plt.yscale('log')
	# plt.show()

	# print(len(energy))
	# fig = plt.figure()
	# plt.plot(hist)
	# plt.show()
	# plotSpectrum(df)

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

	yv, xv, _ = get_hist(myarr, range=(0, 5), dx=0.001) 
	

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
	g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['x']['pages']), columns=['x']), 
                   lsuffix = '_caller', rsuffix = '_other')
	g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['y']['pages']), columns=['y']),
	              lsuffix = '_caller', rsuffix = '_other')
	g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['z']['pages']), columns=['z']),
	              lsuffix = '_caller', rsuffix = '_other')

	print(g4sdf)
	print(g4sntuple['x']['pages'])
	print(type(g4sntuple['x']['pages']))
	exit()

	# apply E cut / detID cut and sum Edeps for each event using loc, groupby, and sum
	# write directly into output dataframe
	detector_hits = g4sdf.loc[(g4sdf.Edep>0)&(g4sdf.volID==1)]
	procdf = pd.DataFrame(detector_hits.groupby(['event','volID','iRep'], as_index=False)['Edep'].sum())
	procdf['x'] = detector_hits.groupby(['event','volID','iRep'], as_index=False)['x']
	procdf['y'] = detector_hits.groupby(['event','volID','iRep'], as_index=False)['y']
	procdf['z'] = detector_hits.groupby(['event','volID','iRep'], as_index=False)['z']
	# procdf = procdf.join(pd.DataFrame(detector_hits.groupby(['event','volID','iRep'], as_index=False)['x']))
	# procdf = procdf.join(pd.DataFrame(detector_hits.groupby(['event','volID','iRep'], as_index=False)['y']))
	# procdf = procdf.join(pd.DataFrame(detector_hits.groupby(['event','volID','iRep'], as_index=False)['z']))
	procdf = procdf.rename(columns={'iRep':'detID', 'Edep':'energy'})

	return procdf


def test2():

    from ROOT import TFile, TTree
    in_file = "out.root"
    tf = TFile(in_file)
    tt = tf.Get("g4sntuple")

    from lat.utils import TDraw
    # g4sntuple->Draw("y[1]:x[1] >> AA(280,-35,35,280,-35,35)","","colz")
    y, x = TDraw(tt, "y[1]:x[1]", "")
    print(len(x))

def test3(filename):
	g4sfile = h5py.File(filename, 'r')
	g4sntuple = g4sfile['default_ntuples']['g4sntuple']
	print(g4sntuple)





if __name__ == '__main__':
	main()