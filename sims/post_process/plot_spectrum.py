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

	# filename = '../alpha/raw_out/ICPC_Pb_241Am_10000000.hdf5'
	# filename = '../alpha/processed_out/processed_ICPC_Pb_241Am_10000000.hdf5'

	# filename = '../alpha/raw_out/test.hdf5'
	# filename = '../alpha/processed_out/processed_test.hdf5'

	# filename = '../alpha/raw_out/sourceRot33_ICPC_Pb_241Am_10000000.hdf5'
	# processed_filename = '../alpha/processed_out/processed_sourceRot33_ICPC_Pb_241Am_10000000.hdf5'


	# filename = '../alpha/raw_out/newDet_sourceRot33_ICPC_Pb_241Am_20000000.hdf5'

	# filename = '../alpha/raw_out/newDet_sourceRotNorm_y6mm_ICPC_Pb_241Am_20000000.hdf5'
	# processed_filename = '../alpha/processed_out/processed_newDet_sourceRot25_thetaDet65_y14mm_ICPC_Pb_241Am_100000000.hdf5'

	# processed_filename = '../alpha/processed_out/oppi/processed_oppi_y19mm_norm_241Am_100000000.hdf5'
	processed_filename = '../alpha/processed_out/oppi/processed_oppi_ring_y10_thetaDet45_241Am_100000000.hdf5'
	# processed_filename = '../alpha/processed_out/oppi/processed_test_oppi_y19mm_norm_241Am_1000000.hdf5'

	# processed_filename = '../alpha/processed_out/processed_newDet_sourceRotNorm_y6mm_ICPC_Pb_241Am_20000000.hdf5'





	# plotHist(processed_filename)
	# post_process(filename, processed_filename, source=False)
	# plotSpot(processed_filename, source=False, plot_title = 'Spot Size from $^{241}$Am (10$^8$ Primaries) \nnormal incidence at 6 mm', particle = 'all')
	# ZplotSpot(filename)
	# spot_curve()
	plot1DSpot(processed_filename, axis='y', particle='all')
	# plot2Dhist(processed_filename, nbins=500, plot_title = 'Spot Size from $^{241}$Am (10$^8$ Primaries) \n 45 deg theta_det at 10 mm', source=False, particle = 'all')
	# plotDepth(processed_filename, source=False, particle = 'all', plot_title='65 deg, 14mm, $10^8$ primaries')
	# plotContour(processed_filename, source=False, particle = 'all')
	# testFit(filename)
	# getCounts(processed_filename)

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



def gauss_fit_func(x, A, mu, sigma, C):
	# return (A * (np.exp(-1.0 * ((x - mu)**2) / (2 * sigma**2))+C) +D)
	# return (A * (np.exp(-1.0 * ((x - mu)**2) / (2 * sigma**2))+C))
	return (A * (1/(sigma*np.sqrt(2*np.pi))) *(np.exp(-1.0 * ((x - mu)**2) / (2 * sigma**2))+C))
	# return (A * (np.exp(-1.0 * ((x - mu)**2) / (2 * sigma**2))))

def kde1D(x, bandwidth=1., bins=100, optimize_bw=True):
	from sklearn.neighbors import KernelDensity
	from sklearn.model_selection import GridSearchCV, LeaveOneOut, KFold, cross_val_score

	cv1 = KFold(n_splits=100)
	cv2 = LeaveOneOut()

	nbins = bins
	data = np.vstack(x)
	if optimize_bw:
		print('optimizing bandwidth of KDE')
		params = {'bandwidth': np.logspace(-2, 1, 80)}
		grid = GridSearchCV(KernelDensity(), params, cv =cv1)
		grid.fit(data)
		bw = grid.best_estimator_.bandwidth
		print('best bandwidth: {0}'.format(grid.best_estimator_.bandwidth))
		score1 = grid.best_score_
		print("score: {0}".format(score1))
	else:
		bw = bandwidth

	print('using bandwidth ', bw)

	score = cross_val_score(KernelDensity(bandwidth=bw), data, cv=cv1)
	print(np.amax(score))

	x_grid = np.linspace(data.min(), data.max(), nbins)
	xi = np.vstack(x_grid)

	kde_skl = KernelDensity(bandwidth=bw)
	kde_skl.fit(data)

	zi = np.exp(kde_skl.score_samples(xi))

	return(x_grid, zi.T, bw)

def kde2D(x, y, bandwidth=1., bins=100, optimize_bw=True):
	from sklearn.neighbors import KernelDensity
	from sklearn.model_selection import GridSearchCV, LeaveOneOut, KFold, cross_val_score

	cv1 = KFold(n_splits=100)
	cv2 = LeaveOneOut()

	nbins = bins
	data_raw = np.vstack([y,x]).T
	if optimize_bw:
		print('optimizing bandwidth of KDE')
		params = {'bandwidth': np.logspace(-2, 1, 80)}
		# kde = KernelDensity().fit(data_raw)
		grid = GridSearchCV(KernelDensity(), params, cv=cv1)
		grid.fit(data_raw)
		bw = grid.best_estimator_.bandwidth
		score1 = grid.best_score_
		print('best score: {0}'.format(grid.best_score_))
		# score = cross_val_score(KernelDensity(bandwidth=bw), data_raw, cv=cv1)
		# score = cross_val_score(KernelDensity(bandwidth=bw), data_raw)

		print('best bandwidth: {0}'.format(grid.best_estimator_.bandwidth))
	else:
		bw = bandwidth

	print('using bandwidth ', bw)

	score = cross_val_score(KernelDensity(bandwidth=bw), data_raw, cv=cv1)
	print(np.amax(score))

	xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
	sample_data = np.vstack([yi.ravel(), xi.ravel()]).T

	kde_skl = KernelDensity(bandwidth=bw)
	kde_skl.fit(data_raw)

	z = np.exp(kde_skl.score_samples(sample_data))
	zi = np.reshape(z, xi.shape)

	return(xi, yi, zi, bw, score)

def getCounts(processed_filename):
	df = pd.read_hdf(processed_filename, keys='procdf')
	energy = np.array(df['energy'])
	counts = len(energy)
	print('%f counts in PV' %counts)

def getCounts_cut(processed_filename, elo, ehi):
	df = pd.read_hdf(processed_filename, keys='procdf')
	energy = np.array(df['energy'])
	cut_df = df.loc[(df.energy > elo) & (df.energy < ehi)]
	cut_energy_keV = np.array(cut_df['energy']*1000)
	counts = len(cut_energy_keV)
	print(f'{counts} counts in region {elo} to {ehi} keV')

	return(counts)

def getRate(processed_filename, primaries, elo, ehi):
	# see this elog https://elog.legend-exp.org/UWScanner/166
	source_activity = 4.0e4 #40 kBq = 4e4 decays/s
	time_seconds = primaries/(source_activity)
	counts = getCounts_cut(processed_filename, elo, ehi)
	rate = counts/time_seconds #(rate in counts/s)
	print(f'{rate} counts/second in region {elo} to {ehi} keV')

	return(rate)

def plotRate(radius, elo, ehi):
	rates_arr = []
	for r in radius:
		rate = getRate(f'../alpha/processed_out/oppi/processed_oppi_ring_y{r}_norm_241Am_100000000.hdf5', 10000000, 5.4, 5.6)
		rates_arr.append(rate)
	plt.plot(radius, rates_arr, '.r')
	plt.show()

def plotHist(filename):
	# df = pandarize(filename)
	pctResAt1MeV = 0.5 #0.5% energy resolution
	df = pd.read_hdf(filename, keys='procdf')
	# apply energy resolution function
	# df['energy'] = df['energy'] + np.sqrt(df['energy'])*pctResAt1MeV/100.*np.random.randn(len(df['energy']))
	energy = np.array(df['energy']*1000)

	# print(energy)
	# exit()
	# pid = np.array(df['pid'])


	# alpha_df = df.loc[df.energy > 5]
	# energy = np.array(alpha_df['energy']*1000)
	# energy = np.array(df['energy']*1000)
	# print(tmp['pid'].astype(int).unique)
	# print(df['pid'].astype(int).unique)
	# exit()
	# x = np.array(df['x'])
	# y = np.array(df['y'])
	# z = np.array(df['z'])

	# x = np.array(alpha_df['x'])
	# y = np.array(alpha_df['y'])
	# z = np.array(alpha_df['z'])

	# fig, ax = plt.subplots()
	fig, ax = plt.subplots(figsize=(10,8))
	plt.hist(energy, range = [0, 100], bins=200)
	plt.yscale('log')
	ax.set_xlabel('Energy (keV)', fontsize=16)
	ax.set_ylabel('Counts/2 keV', fontsize=16)
	plt.setp(ax.get_xticklabels(), fontsize=14)
	plt.setp(ax.get_yticklabels(), fontsize=14)
	# plt.title('Collimated, $^{241}$Am 7*10$^5$ Primaries, 16 mm above detector', fontsize=18)
	plt.title('$^{241}$Am 10$^8$ Primaries, normal incidence at 9 mm, 0.5 keV bins, assuming 0.5\% resolution', fontsize=18)
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

def plot2Dhist(filename, nbins=100, plot_title = '', source=False, particle = 'all'):

	df = pd.read_hdf(filename, keys='procdf')

	if particle == 'all':
		x = np.array(df['x'])
		y = np.array(df['y'])
		z = np.array(df['z'])
		energy = np.array(df['energy']*1000)
		# plot_title = 'Spot Size, $^{241}$Am 10$^7$ Primaries, all energies'

	elif particle == 'alpha':
		alpha_df = df.loc[df.energy > 5]
		x = np.array(alpha_df['x'])
		y = np.array(alpha_df['y'])
		z = np.array(alpha_df['z'])
		energy = np.array(alpha_df['energy']*1000)
		# plot_title = 'Spot Size, $^{241}$Am 10$^7$ Primaries, Energy $>$ 5 MeV'

	elif particle == 'gamma':
		gamma_df = df.loc[(df.energy > .04) & (df.energy < 0.08)]
		x = np.array(gamma_df['x'])
		y = np.array(gamma_df['y'])
		z = np.array(gamma_df['z'])
		energy = np.array(gamma_df['energy']*1000)
		# plot_title = 'Spot Size, $^{241}$Am 10$^7$ Primaries, 60 kev $<$ Energy $<$ 80 keV'

	else:
		print('specify particle type!')
		exit()


	fig, ax = plt.subplots(figsize=(10,8))
	nbins=nbins

	# hist = ax.hist2d(x, y, bins=nbins, cmap='plasma', normed=True)
	hist = ax.hist2d(x, y, bins=nbins, cmap='plasma', norm=LogNorm())
	cb = plt.colorbar(hist[3], ax=ax)
	cb.set_label("Counts", ha = 'right', va='bottom', rotation=270, fontsize=28)
	# plt.xlim(-40,40)
	# plt.ylim(-40,40)
	# ax.set_xlabel('x position (mm)', fontsize=16)
	# ax.set_ylabel('y position (mm)', fontsize=16)
	# plt.setp(ax.get_xticklabels(), fontsize=14)
	# plt.setp(ax.get_yticklabels(), fontsize=14)
	# plt.title(plot_title, fontsize=16)

	cb.ax.tick_params(labelsize=20)
	plt.xlim(-40,40)
	plt.ylim(-40,40)
	ax.set_xlabel('x position (mm)', fontsize=28)
	ax.set_ylabel('y position (mm)', fontsize=28)
	plt.setp(ax.get_xticklabels(), fontsize=20)
	plt.setp(ax.get_yticklabels(), fontsize=20)
	plt.title(plot_title, fontsize=32)
	plt.show()

def plotContour(filename, source=False, particle = 'all'):

	df = pd.read_hdf(filename, keys='procdf')

	if particle == 'all':
		x = np.array(df['x'])
		y = np.array(df['y'])
		z = np.array(df['z'])
		energy = np.array(df['energy']*1000)
		plot_title = 'Spot Size, $^{241}$Am 10$^7$ Primaries, all energies'

	elif particle == 'alpha':
		alpha_df = df.loc[df.energy > 5]
		x = np.array(alpha_df['x'])
		y = np.array(alpha_df['y'])
		z = np.array(alpha_df['z'])
		energy = np.array(alpha_df['energy']*1000)
		plot_title = 'Spot Size, $^{241}$Am 10$^7$ Primaries, Energy $>$ 5 MeV'

	elif particle == 'gamma':
		gamma_df = df.loc[(df.energy > .04) & (df.energy < 0.08)]
		x = np.array(gamma_df['x'])
		y = np.array(gamma_df['y'])
		z = np.array(gamma_df['z'])
		energy = np.array(gamma_df['energy']*1000)
		plot_title = 'Spot Size, $^{241}$Am 10$^7$ Primaries, 60 kev $<$ Energy $<$ 80 keV'

	else:
		print('specify particle type!')
		exit()


	# fig, ax = plt.subplots(ncols=3, figsize=(16,8))
	fig, ax = plt.subplots(figsize=(10,8))
	nbins=100
	counts, xbins, ybins = np.histogram2d(x, y, bins=nbins, normed=True)
	# hist = ax[0].hist2d(x, y, bins=nbins, cmap='plasma', normed=True)
	hist = ax.hist2d(x, y, bins=nbins, cmap='plasma', normed=True)
	# plt.scatter(x, y, c=energy, s=1, cmap='plasma')
	# cb = plt.colorbar()
	# cb.set_label("Energy (keV)", ha = 'right', va='center', rotation=270, fontsize=14)
	# cb.ax.tick_params(labelsize=12)
	# ax[0].set_xlim(-10,10)
	# ax[0].set_ylim(9,19)
	# ax[0].set_xlabel('x position (mm)', fontsize=14)
	# ax[0].set_ylabel('y position (mm)', fontsize=14)
	# ax[0].set_title('Histogram of data- 100 bins', fontsize=14)

	# ax.set_xlim(-10,10)
	# ax.set_ylim(-10, 10)
	# # ax.set_ylim(9,19)
	# ax.set_xlabel('x position (mm)', fontsize=14)
	# ax.set_ylabel('y position (mm)', fontsize=14)
	# ax.set_title('Histogram of data- 100 bins', fontsize=14)

	# CB2 = plt.colorbar(hist[3], shrink=0.8, extend='both')

	# xi, yi, zi, bw, score = kde2D(x, y, bins=500, optimize_bw=True)
	# xi, yi, zi, bw, score = kde2D(x, y, bandwidth=0.25, bins=500, optimize_bw=False)
	# x_score = np.linspace(0, len(score), 200)

	# fig, ax = plt.subplots(figsize=(10,8))
	# plt.plot(score)
	# ax.hist(score, bins=100)
	# plt.show()

	# exit()




	# kdeColor = ax[1].pcolormesh(xi, yi, zi, cmap='plasma')
	# ax[1].pcolormesh(xi, yi, norm_zi.reshape(xi.shape), cmap='plasma')
	# ax[1].set_xlim(-10,10)
	# ax[1].set_ylim(9,19)
	# ax[1].set_xlabel('x position (mm)', fontsize=14)
	# ax[1].set_ylabel('y position (mm)', fontsize=14)
	# ax[1].set_title('KDE-smoothed \n Bandwidth = %.2f' % bw, fontsize=14)
	# CB1 = plt.colorbar(kdeColor, shrink=0.8, extend='both')

	# levels = [0.1]

	# contour_hist = ax[2].contour(counts.T,extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],cmap='plasma')

	# CS = ax[2].contour(xi, yi, zi, cmap='plasma')

	# ax[2].clabel(CS, fmt = '%.2f', fontsize=14)
	# CB = plt.colorbar(CS, shrink=0.8, extend='both')
	# ax[2].clabel(contour_hist, fmt = '%.2f', fontsize=20)
	# CB = plt.colorbar(contour_hist, shrink=0.8, extend='both')

	# ax[2].set_xlim(-10,10)
	# ax[2].set_ylim(9,19)
	# ax[2].set_xlabel('x position (mm)', fontsize=14)
	# ax[2].set_ylabel('y position (mm)', fontsize=14)
	# ax[2].set_title('Contour plot from KDE', fontsize=14)
	# ax[2].set_title('Contour plot from histogram', fontsize=14)
	# CB = plt.colorbar(contour_hist, shrink=0.8, extend='both')
	# ax[2].clabel(contour_hist, fmt = '%.2f', fontsize=20)


	# plt.xlim(-40,40)
	# plt.ylim(-40,40)
	# ax[0].set_xlabel('x position (mm)', fontsize=16)
	# ax[0].set_ylabel('y position (mm)', fontsize=16)
	# plt.setp(ax[0].get_xticklabels(), fontsize=14)
	# plt.setp(ax[0].get_yticklabels(), fontsize=14)
	# plt.title(plot_title, fontsize=16)
	plt.show()

def old_plotContour(filename, source=False, particle = 'all'):

	df = pd.read_hdf(filename, keys='procdf')

	if particle == 'all':
		x = np.array(df['x'])
		y = np.array(df['y'])
		z = np.array(df['z'])
		energy = np.array(df['energy']*1000)
		plot_title = 'Spot Size, $^{241}$Am 10$^7$ Primaries, all energies'

	elif particle == 'alpha':
		alpha_df = df.loc[df.energy > 5]
		x = np.array(alpha_df['x'])
		y = np.array(alpha_df['y'])
		z = np.array(alpha_df['z'])
		energy = np.array(alpha_df['energy']*1000)
		plot_title = 'Spot Size, $^{241}$Am 10$^7$ Primaries, Energy $>$ 5 MeV'

	elif particle == 'gamma':
		gamma_df = df.loc[(df.energy > .04) & (df.energy < 0.08)]
		x = np.array(gamma_df['x'])
		y = np.array(gamma_df['y'])
		z = np.array(gamma_df['z'])
		energy = np.array(gamma_df['energy']*1000)
		plot_title = 'Spot Size, $^{241}$Am 10$^7$ Primaries, 60 kev $<$ Energy $<$ 80 keV'

	else:
		print('specify particle type!')
		exit()


	fig, ax = plt.subplots(ncols=3)
	nbins=10
	counts, xbins, ybins = np.histogram2d(x, y, bins=nbins, normed=True)
	ax[0].hist2d(x, y, bins=nbins, cmap='plasma', normed=True)
	# plt.scatter(x, y, c=energy, s=1, cmap='plasma')
	# cb = plt.colorbar()
	# cb.set_label("Energy (keV)", ha = 'right', va='center', rotation=270, fontsize=14)
	# cb.ax.tick_params(labelsize=12)
	ax[0].set_xlim(-10,10)
	ax[0].set_ylim(9,19)
	ax[0].set_xlabel('x position (mm)', fontsize=14)
	ax[0].set_ylabel('y position (mm)', fontsize=14)
	ax[0].set_title('2D histogam- 10 bins', fontsize=14)

	# k_arr = np.column_stack((x,y))
	# k = kde.gaussian_kde(k_arr.T)
	xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
	# zi = k(np.vstack([xi.flatten(), yi.flatten()]))
	positions = np.vstack([xi.flatten(), yi.flatten()])
	values = np.vstack([x,y])
	kernel = kde.gaussian_kde(values)
	zi = np.reshape(kernel(positions).T, xi.shape)
	print(np.sum(zi))
	scale = len(x)/np.sum(zi)
	zi *= scale
	# print(np.sum(counts))
	# print(np.min(zi), np.max(zi))
	# exit()

	# norm = np.linalg.norm(zi)
	# norm_zi = zi/norm
	print(zi)
	# print(xi.flatten())
	exit()
	# exit()
	# ax[1].pcolormesh(xi, yi, zi.reshape(xi.shape), cmap='plasma')
	ax[1].pcolormesh(xi, yi, zi, cmap='plasma')
	# ax[1].pcolormesh(xi, yi, norm_zi.reshape(xi.shape), cmap='plasma')
	ax[1].set_xlim(-10,10)
	ax[1].set_ylim(9,19)
	ax[1].set_xlabel('x position (mm)', fontsize=14)
	ax[1].set_ylabel('y position (mm)', fontsize=14)
	ax[1].set_title('KDE-smoothed', fontsize=14)

	levels = [0.1]

	contour_hist = ax[2].contour(counts.T,extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],cmap='plasma')
	# CS = ax[2].contour(xi, yi, zi.reshape(xi.shape), cmap='plasma')
	# CS = ax[2].contour(xi, yi, zi, cmap='plasma')
	# CSF = ax[2].contourf(xi, yi, norm_zi.reshape(xi.shape), cmap='plasma')
	# CSF = ax[2].contourf(xi, yi, zi.reshape(xi.shape), cmap='plasma')
	# plt.clabel(CS, fmt = '%2.1d', colors = 'k', fontsize=14)
	# ax[2].clabel(CS, fmt = '%.2f', fontsize=20)
	# CB = plt.colorbar(CS, shrink=0.8, extend='both')
	# ax[2].clabel(contour_hist, fmt = '%.2f', fontsize=20)
	CB = plt.colorbar(contour_hist, shrink=0.8, extend='both')

	ax[2].set_xlim(-10,10)
	ax[2].set_ylim(9,19)
	ax[2].set_xlabel('x position (mm)', fontsize=14)
	ax[2].set_ylabel('y position (mm)', fontsize=14)
	# ax[2].set_title('Contour plot from KDE', fontsize=14)
	ax[2].set_title('Contour plot from histogram', fontsize=14)
	# CB = plt.colorbar(contour_hist, shrink=0.8, extend='both')
	# ax[2].clabel(contour_hist, fmt = '%.2f', fontsize=20)


	# plt.xlim(-40,40)
	# plt.ylim(-40,40)
	# ax[0].set_xlabel('x position (mm)', fontsize=16)
	# ax[0].set_ylabel('y position (mm)', fontsize=16)
	# plt.setp(ax[0].get_xticklabels(), fontsize=14)
	# plt.setp(ax[0].get_yticklabels(), fontsize=14)
	# plt.title(plot_title, fontsize=16)
	plt.show()

	if source==True:
		source_df = pd.read_hdf(filename, keys='sourcePV_df')
		sourceEnergy = np.array(source_df['energy']*1000)
		x_source = np.array(source_df['x'])
		print(len(x_source))

def plotDepth(filename, plot_title, source=False, particle = 'all'):

	df = pd.read_hdf(filename, keys='procdf')
	df = df.loc[(df.x > -0.25) & (df.x < 0.25) & df.energy > 0.01]


	if particle == 'all':
		x = np.array(df['x'])
		y = np.array(df['y'])
		z = np.array(df['z'])
		energy = np.array(df['energy']*1000)

	elif particle == 'alpha':
		alpha_df = df.loc[df.energy > 5]
		x = np.array(alpha_df['x'])
		y = np.array(alpha_df['y'])
		z = np.array(alpha_df['z'])
		energy = np.array(alpha_df['energy']*1000)

	elif particle == 'gamma':
		gamma_df = df.loc[(df.energy > .04) & (df.energy < 0.08)]
		x = np.array(gamma_df['x'])
		y = np.array(gamma_df['y'])
		z = np.array(gamma_df['z'])
		energy = np.array(gamma_df['energy']*1000)


	else:
		print('specify particle type!')
		exit()

	z = z + 22.5 #make surface at 0 z depth


	fig, ax = plt.subplots(figsize=(9,8))
	# plot_title = 'Spot Size from $^{241}$Am, 10$^7$ Primaries'
	plt.scatter(y, z, c=energy, s=1, cmap='plasma', norm=LogNorm(10,6000))
	# plt.scatter(x, y, c=energy, s=1, cmap='plasma')
	cb = plt.colorbar()
	cb.set_label("Energy (keV)", ha = 'right', va='center', rotation=270, fontsize=20)
	cb.ax.tick_params(labelsize=18)
	plt.xlim(-31,31)
	# plt.ylim(-0.05,0.01)
	# plt.xlim(8,21)
	plt.ylim(-2.5,0.5)
	ax.set_xlabel('y position (mm)', fontsize=20)
	ax.set_ylabel('z position (mm)', fontsize=20)
	plt.setp(ax.get_xticklabels(), fontsize=18)
	plt.setp(ax.get_yticklabels(), fontsize=18)
	plt.title('Depth for: ' + plot_title, fontsize=20)
	plt.show()

	if source==True:
		source_df = pd.read_hdf(filename, keys='sourcePV_df')
		sourceEnergy = np.array(source_df['energy']*1000)
		x_source = np.array(source_df['x'])
		print(len(x_source))

def plotSpot(filename, source=False, plot_title = '', particle = 'all'):

	df = pd.read_hdf(filename, keys='procdf')

	if particle == 'all':
		x = np.array(df['x'])
		y = np.array(df['y'])
		z = np.array(df['z'])
		energy = np.array(df['energy']*1000)
		# plot_title = 'Spot Size, $^{241}$Am 10$^7$ Primaries'

	elif particle == 'alpha':
		alpha_df = df.loc[df.energy > 5]
		x = np.array(alpha_df['x'])
		y = np.array(alpha_df['y'])
		z = np.array(alpha_df['z'])
		energy = np.array(alpha_df['energy']*1000)
		# plot_title = 'Spot Size from $^{241}$Am, 10$^7$ Primaries, Energy $>$ 5 MeV'

	elif particle == 'gamma':
		gamma_df = df.loc[(df.energy > .04) & (df.energy < 0.08)]
		x = np.array(gamma_df['x'])
		y = np.array(gamma_df['y'])
		z = np.array(gamma_df['z'])
		energy = np.array(gamma_df['energy']*1000)
		# plot_title = 'Spot Size from $^{241}$Am, 10$^7$ Primaries'

	else:
		print('specify particle type!')
		exit()


	fig, ax = plt.subplots(figsize=(11,9))
	# plot_title = 'Spot Size from $^{241}$Am, 10$^7$ Primaries'
	plt.scatter(x, y, c=energy, s=1, cmap='plasma', norm=LogNorm(10,6000))
	# plt.scatter(x, y, c=energy, s=1, cmap='plasma')
	cb = plt.colorbar()
	cb.set_label("Energy (keV)", ha = 'right', va='bottom', rotation=270, fontsize=28)
	cb.ax.tick_params(labelsize=20)
	plt.xlim(-40,40)
	plt.ylim(-40,40)
	ax.set_xlabel('x position (mm)', fontsize=28)
	ax.set_ylabel('y position (mm)', fontsize=28)
	plt.setp(ax.get_xticklabels(), fontsize=20)
	plt.setp(ax.get_yticklabels(), fontsize=20)
	plt.title(plot_title, fontsize=32)
	plt.show()

	if source==True:
		source_df = pd.read_hdf(filename, keys='sourcePV_df')
		sourceEnergy = np.array(source_df['energy']*1000)
		x_source = np.array(source_df['x'])
		print(len(x_source))


def plot1DSpot(filename, axis='x', particle = 'all', fit=True):
	axis = str(axis)
	particle = str(particle)
	df = pd.read_hdf(filename, keys='procdf')
	energy = np.array(df['energy'])

	if particle=='all':
		scale_std = 1.
		if axis=='x':
			x = np.array(df['x'])
		elif axis=='y':
			x = np.array(df['y'])
		else:
			print('Specify fit axis! Can be x or y')
			exit()

	elif particle=='alpha':
		alpha_df = df.loc[df.energy > 5]
		scale_std = 1.
		if axis=='x':
			x = np.array(alpha_df['x'])
		elif axis=='y':
			x = np.array(alpha_df['y'])
		else:
			Print('Specify fit axis! Can be x or y')
			exit()

	elif particle=='gamma':
		gamma_df = df.loc[(df.energy > .04) & (df.energy < 0.08)]
		scale_std = 1.
		if axis=='x':
			x = np.array(gamma_df['x'])
		elif axis=='y':
			x = np.array(gamma_df['y'])
		else:
			print('Specify fit axis! Can be x or y')
			exit()
	else:
		print('Specify particle type! Can be: all, alpha, or gamma ')
		exit()



	mean = np.mean(x)
	median = np.median(x)
	std = np.std(x)
	amin = np.amin(x)
	amax = np.amax(x)
	minvalue = int(amin)
	maxvalue = int(amax)
	mom_mean = scipy.stats.moment(x, moment=1)
	mom_var = scipy.stats.moment(x, moment=2)
	mom_skew = scipy.stats.moment(x, moment=3)

	print('median: ', median, ' std: ', std, ' mean: ', mean)
	print('moment mean: ', mom_mean, 'moment variance: ', mom_var, 'moment_skew: ', mom_skew)

	# x1, y1, bw = kde1D(x, bins=500, bandwidth=0.25, optimize_bw=False)
	x1, y1, bw = kde1D(x, bins=500, optimize_bw=True)
	std_kde = np.std(y1)
	print(std_kde)
	# exit()


	#Fit the kde-smoothed histogram
	popt, pcov = curve_fit(gauss_fit_func, xdata=x1, ydata=y1, p0=[1, median, std, 1])
	perr = np.sqrt(np.abs(np.diag(pcov))) # Variances of each fit parameter
	print(perr)
	# print(pcov)
	# print(type(popt))

	print('Amplitude: ', popt[0], ' Fit mean = ', popt[1], ' Fit Stdv = ', popt[2], 'C = ', popt[3])
	# print('mean = ', popt[1], 'stdev = ', popt[2])
	# print('popt: ', popt)

	sigma = np.abs(popt[2])
	fwhm = sigma*2.355
	array_fwhm = std*2.355

	sigma_uncertainty = perr[2]
	print('sigma = ', sigma, '+/- ', sigma_uncertainty)
	sigma_percentUncertainty = (sigma_uncertainty/sigma)*100
	print('sigma percent uncertainty ', sigma_percentUncertainty)

	print('FWHM = ', fwhm, '+/- ', 2.355*sigma_uncertainty)

	print('FWHM = ', fwhm,  '; array FWHM = ', array_fwhm)

	fig, ax = plt.subplots(figsize=(10,8))
	# ax.figure(figsize=(10,10))

	ax.hist(x, bins = 100, density=True, color='grey', alpha=0.75, label = 'Normalized histogram \nof data: %.f entries \n100 bins' % len(x))
	# ax.hist(xfit, bins = bins, label = 'Data: %.f entries \n 0.25 mm bins' % len(xfit))

	# ax.plot(x1, y1, 'k-', linewidth=3, label='Gaussian KDE of data: %.2f bandwidth' % bw)

	# ax.plot(x1, gauss_fit_func(x1, *popt), 'r--', linewidth = 2, label='Gaussian Fit of KDE: FWHM = %.2f mm' % fwhm)
	ax.plot(x1, gauss_fit_func(x1, *popt), 'r--', linewidth = 2, label='Gaussian Fit: \nFWHM = %.2f mm' % fwhm)
	legend = ax.legend(loc='upper right', shadow = False, fontsize='20')



	ax.set_xlabel('y position (mm)', fontsize=28)

	plt.setp(ax.get_xticklabels(), fontsize=20)

	plt.setp(ax.get_yticklabels(), fontsize=20)
	plt.xlim(-5,5)

	# plt.title('Y-axis projection of spot-size. Gaussian KDE, %.2f bandwidth' % bw, fontsize=16)
	plt.title('Y-axis projection of spot-size with Gaussian fit', fontsize=35)

	plt.show()

def spot_curve():
	angles = [90, 75, 65, 55, 45]
	spotSize = [1.98, 2.08, 2.37, 2.87, 3.79]
	uncertainty = [0.00569, 0.00570, 0.0066, 0.00784, 0.0118]

	fig, ax = plt.subplots(figsize=(6,5))

	plt.plot(angles, spotSize, '.r')

	plt.xlabel('Theta Det (deg)', fontsize=16)
	plt.ylabel('Spot size (mm)', fontsize=16)
	plt.setp(ax.get_xticklabels(), fontsize=14)
	plt.setp(ax.get_yticklabels(), fontsize=14)
	plt.title('Y-axis projection of spot-size for various source angles', fontsize=16)

	plt.show()

def rate_curve():
	radii = [19., 22., 25., 28., 31.]


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
