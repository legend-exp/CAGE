import numpy as np
import scipy
import matplotlib
from matplotlib.colors import LogNorm
from scipy.stats import norm
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import ROOT
import sys
from particle import PDGID
matplotlib.rcParams['text.usetex'] = True

def main():

	# data_dir = '../alpha/raw_out/'
	# file = 'test_e100000.hdf5'
	# filename = data_dir + file

	# filename = '../alpha/raw_out/uncollimated_test_241Am_700000.hdf5'
	# filename = '../alpha/raw_out/collimated_test_241Am_700000.hdf5'
	# filename = '../alpha/raw_out/30mm_collimated_241Am_700000.hdf5'
	# filename = '../alpha/raw_out/30mm_notcollimated_241Am_700000.hdf5'
	# filename = '../alpha/processed_out/processed_30mm_notcollimated_241Am_700000.hdf5'
	# filename = '../alpha/raw_out/30mm_collimated_241Am_10000000.hdf5'
	# filename = '../alpha/processed_out/processed_30mm_collimated_241Am_10000000.hdf5'
	# filename = '../alpha/raw_out/22mm_collimated_241Am_10000000.hdf5'
	# filename = '../alpha/raw_out/16mm_collimated_241Am_10000000.hdf5'

	# filename = '../alpha/processed_out/processed_22mm_collimated_241Am_10000000.hdf5'
	# filename = '../alpha/processed_out/processed_16mm_collimated_241Am_10000000.hdf5'

	# filename = '../alpha/raw_out/noDet_Cu_22mm_collimated_241Am_100000.hdf5'
	# filename = '../alpha/processed_out/processed_noDet_Cu_22mm_collimated_241Am_100000.hdf5'

	# filename = '../alpha/raw_out/noDet_Cu_22mm_collimated_241Am_10000000.hdf5'
	# filename = '../alpha/processed_out/processed_noDet_Cu_22mm_collimated_241Am_10000000.hdf5'


	# filename = '../alpha/raw_out/newTest_Pb_241Am_1000000.hdf5'
	# filename = '../alpha/processed_out/processed_newTest_Pb_241Am_1000000.hdf5'
	# filename = '../alpha/raw_out/topHat_Test_Pb_241Am_1000000.hdf5'
	# filename = '../alpha/processed_out/processed_topHat_Test_Pb_241Am_1000000.hdf5'


	# filename = '../alpha/raw_out/topHat_Test2_Pb_241Am_1000000.hdf5'
	filename = '../alpha/processed_out/processed_topHat_Test2_Pb_241Am_1000000.hdf5'



	#filename = '../alpha/raw_out/test_sebColl_e100000.hdf5'

	# plotHist(filename)
	# post_process(filename)
	# plotSpot(filename)
	ZplotSpot(filename)
	# plot1DSpot(filename)
	# testFit(filename)




	# pandarize(filename)
	# test_func(filename)
	# exit()

	# test3(filename)
	# test1(filename)
	# test2()
	# test4(filename)

# def test4(filename):

def post_process(filename):
	df = pandarize(filename)
	# df.to_hdf('../alpha/processed_out/processed_30mm_notcollimated_241Am_700000.hdf5', key='procdf', mode='w')
	df.to_hdf('../alpha/processed_out/processed_topHat_Test2_Pb_241Am_1000000.hdf5', key='procdf', mode='w')


def plotHist(filename):
	# df = pandarize(filename)
	df = pd.read_hdf(filename, keys='procdf')
	energy = np.array(df['energy'])
	# print(energy)
	# exit()
	# pid = np.array(df['pid'])


	# alpha_df = df.loc[df.energy > 5]
	# energy = np.array(alpha_df['energy']*1000)
	energy = np.array(df['energy']*1000)
	# print(tmp['pid'].astype(int).unique)
	# print(df['pid'].astype(int).unique)
	# exit()
	# x = np.array(df['x'])
	# y = np.array(df['y'])
	# z = np.array(df['z'])

	# x = np.array(alpha_df['x'])
	# y = np.array(alpha_df['y'])
	# z = np.array(alpha_df['z'])

	fig, ax = plt.subplots()
	plt.hist(energy, range = [0.0, 6000], bins=600)
	plt.yscale('log')
	ax.set_xlabel('Energy (keV)', fontsize=16)
	ax.set_ylabel('Counts/10 keV', fontsize=16)
	plt.setp(ax.get_xticklabels(), fontsize=14)
	plt.setp(ax.get_yticklabels(), fontsize=14)
	# plt.title('Collimated, $^{241}$Am 7*10$^5$ Primaries, 16 mm above detector', fontsize=18)
	plt.title('$^{241}$Am 10$^7$ Primaries, Coll. 22 mm above detector (no E-res func)', fontsize=18)
	plt.show()

def ZplotSpot(filename):
	# df = pandarize(filename)
	df = pd.read_hdf(filename, keys='procdf')
	energy = np.array(df['energy']*1000)
	# alpha_df = df.loc[df.energy > 5]
	# gamma_df = df.loc[(df.energy > .04) & (df.energy > 0.08)]

	x = np.array(df['x'])
	y = np.array(df['y'])
	z = np.array(df['z'])
	# x = np.array(alpha_df['x'])
	# y = np.array(alpha_df['y'])
	# z = np.array(alpha_df['z'])
	# x = np.array(gamma_df['x'])
	# y = np.array(gamma_df['y'])
	# z = np.array(gamma_df['z'])


	# energy = np.array(alpha_df['energy']*1000)
	# energy = np.array(gamma_df['energy']*1000)
	energy = np.array(df['energy']*1000)

	fig, ax = plt.subplots()
	# spot_hist = ax.hist2d(x, y, range = [[-32., 32.],[-32., 32.]], weights=energy, norm=LogNorm(), bins=6000) #, range = [[-20., 20.],[-20., 20.]]
	# spot_hist = ax.hist2d(x, y, range = [[-32., 32.],[-32., 32.]], norm=LogNorm(), bins=6000) #, range = [[-20., 20.],[-20., 20.]]
	# plt.colorbar(spot_hist[3], ax=ax)
	# plt.title('Collimated, $^{241}$Am 7*10$^5$ Primaries, 16 mm above detector', fontsize=18)


	# plt.scatter(x, y, c=energy, s=1, cmap='plasma', norm=LogNorm(1,6000))
	plt.scatter(y, z, c=energy, s=1, cmap='plasma')
	cb = plt.colorbar()
	cb.set_label("Energy (keV)", ha = 'right', va='center', rotation=270, fontsize=14)
	cb.ax.tick_params(labelsize=12)
	plt.xlim(-100,100)
	plt.ylim(-100,100)
	ax.set_xlabel('x position (mm)', fontsize=16)
	ax.set_ylabel('z position (mm)', fontsize=16)
	plt.setp(ax.get_xticklabels(), fontsize=14)
	plt.setp(ax.get_yticklabels(), fontsize=14)
	# plt.title('Spot Size, $^{241}$Am 10$^7$ Primaries, Coll. 22 mm above detector, energy 40-80 keV', fontsize=16)
	plt.title('Spot Size, $^{241}$Am 10$^6$ Primaries', fontsize=16)
	plt.show()

def plotSpot(filename):
	# df = pandarize(filename)
	df = pd.read_hdf(filename, keys='procdf')
	energy = np.array(df['energy']*1000)
	# alpha_df = df.loc[df.energy > 5]
	# gamma_df = df.loc[(df.energy > .04) & (df.energy > 0.08)]

	x = np.array(df['x'])
	y = np.array(df['y'])
	z = np.array(df['z'])
	# x = np.array(alpha_df['x'])
	# y = np.array(alpha_df['y'])
	# z = np.array(alpha_df['z'])
	# x = np.array(gamma_df['x'])
	# y = np.array(gamma_df['y'])
	# z = np.array(gamma_df['z'])


	# energy = np.array(alpha_df['energy']*1000)
	# energy = np.array(gamma_df['energy']*1000)
	energy = np.array(df['energy']*1000)

	fig, ax = plt.subplots()
	# spot_hist = ax.hist2d(x, y, range = [[-32., 32.],[-32., 32.]], weights=energy, norm=LogNorm(), bins=6000) #, range = [[-20., 20.],[-20., 20.]]
	# spot_hist = ax.hist2d(x, y, range = [[-32., 32.],[-32., 32.]], norm=LogNorm(), bins=6000) #, range = [[-20., 20.],[-20., 20.]]
	# plt.colorbar(spot_hist[3], ax=ax)
	# plt.title('Collimated, $^{241}$Am 7*10$^5$ Primaries, 16 mm above detector', fontsize=18)


	# plt.scatter(x, y, c=energy, s=1, cmap='plasma', norm=LogNorm(1,6000))
	plt.scatter(x, y, c=energy, s=1, cmap='plasma')
	cb = plt.colorbar()
	cb.set_label("Energy (keV)", ha = 'right', va='center', rotation=270, fontsize=14)
	cb.ax.tick_params(labelsize=12)
	plt.xlim(-40,40)
	plt.ylim(-40,40)
	ax.set_xlabel('x position (mm)', fontsize=16)
	ax.set_ylabel('y position (mm)', fontsize=16)
	plt.setp(ax.get_xticklabels(), fontsize=14)
	plt.setp(ax.get_yticklabels(), fontsize=14)
	# plt.title('Spot Size, $^{241}$Am 10$^7$ Primaries, Coll. 22 mm above detector, energy 40-80 keV', fontsize=16)
	plt.title('Spot Size, $^{241}$Am 10$^6$ Primaries', fontsize=16)
	plt.show()

def plot1DSpot(filename):
	# df = pandarize(filename)
	df = pd.read_hdf(filename, keys='procdf')
	energy = np.array(df['energy'])
	alpha_df = df.loc[df.energy > 5]
	gamma_df = df.loc[(df.energy > .04) & (df.energy > 0.06)]

	# x = np.array(df['x'])
	# y = np.array(df['y'])
	# z = np.array(df['z'])
	x = np.array(alpha_df['x'])
	y = np.array(alpha_df['y'])
	z = np.array(alpha_df['z'])

	# energy = np.array(alpha_df['energy']*1000)
	energy = np.array(gamma_df['energy']*1000)

	(mu, sigma) = norm.fit(x)
	fwhm = sigma*2.355

	fig, ax = plt.subplots()
	plt.hist(x, bins=100)
	y = scipy.stats.norm.pdf(100, mu, sigma)
	l = plt.plot(100, y, 'r--', linewidth=2)
	# spot_hist = ax.hist2d(x, y, range = [[-32., 32.],[-32., 32.]], weights=energy, norm=LogNorm(), bins=6000) #, range = [[-20., 20.],[-20., 20.]]
	# spot_hist = ax.hist2d(x, y, range = [[-32., 32.],[-32., 32.]], norm=LogNorm(), bins=6000) #, range = [[-20., 20.],[-20., 20.]]
	# plt.colorbar(spot_hist[3], ax=ax)
	# plt.title('Collimated, $^{241}$Am 7*10$^5$ Primaries, 16 mm above detector', fontsize=18)


	# plt.scatter(x, y, c=energy, s=1, cmap='plasma', norm=LogNorm(1,6000))
	# plt.scatter(x, y, c=energy, s=1, cmap='plasma')
	# cb = plt.colorbar()
	# cb.set_label("Energy (keV)", ha = 'right', va='center', rotation=270, fontsize=14)
	# cb.ax.tick_params(labelsize=12)
	plt.xlim(-31,31)
	# plt.ylim(-31,31)
	ax.set_xlabel('x position (mm)', fontsize=14)
	# ax.set_ylabel('y position (mm)', fontsize=14)
	plt.setp(ax.get_xticklabels(), fontsize=12)
	plt.setp(ax.get_yticklabels(), fontsize=12)
	plt.title('Spot Size, $^{241}$Am 10$^7$ Primaries, Coll. 22 mm above detector, energy $>$ 5 MeV. FWHM: %.2f mm' % fwhm, fontsize=14)
	print(mu, ', ', sigma)
	print('FWHM: ', sigma*2.355)
	plt.show()


def gauss(xdata, mu, sigma, a):
	return(a*np.exp(((xdata-mu)/sigma)**2))

def testFit(filename):
	df = pd.read_hdf(filename, keys='procdf')
	xbins = np.linspace(0,6000,num=600)
	energy = np.array(df['energy']*1000)
	x = np.array(alpha_df['x'])

	pop, pcov = curve_fit(gauss, xbins, x)



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


if __name__ == '__main__':
	main()
