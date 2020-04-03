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


	# filename = '../alpha/raw_out/ICPC_Pb_241Am_10000000.hdf5'
	# filename = '../alpha/processed_out/processed_ICPC_Pb_241Am_10000000.hdf5'
	# filename = '../alpha/raw_out/test.hdf5'
	# filename = '../alpha/processed_out/processed_test.hdf5'

	# filename = '../alpha/raw_out/sourceRot33_ICPC_Pb_241Am_10000000.hdf5'
	filename = '../alpha/processed_out/processed_sourceRot33_ICPC_Pb_241Am_10000000.hdf5'



	#filename = '../alpha/raw_out/test_sebColl_e100000.hdf5'

	# plotHist(filename)
	# post_process(filename, source=False)
	plotSpot(filename, source=False, particle = 'alpha')
	# ZplotSpot(filename)
	# plot1DSpot(filename, axis='x', particle='alpha')
	# plotContour(filename, source=False, particle = 'alpha')
	# testFit(filename)




	# pandarize(filename)
	# test_func(filename)
	# exit()

	# test3(filename)
	# test1(filename)
	# test2()
	# test4(filename)

# def test4(filename):

def post_process(filename, source=False):
	if source==True:
		procdf, sourcePV_df = pandarize(filename, source)
		# df.to_hdf('../alpha/processed_out/processed_30mm_notcollimated_241Am_700000.hdf5', key='procdf', mode='w')
		procdf.to_hdf('../alpha/processed_out/processed_sourceRot33_ICPC_Pb_241Am_10000000.hdf5', key='procdf', mode='w')
		sourcePV_df.to_hdf('../alpha/processed_out/processed_sourceRot33_ICPC_Pb_241Am_10000000.hdf5', key='sourcePV_df', mode='w')

	else:
		procdf = pandarize(filename, source)
		# df.to_hdf('../alpha/processed_out/processed_30mm_notcollimated_241Am_700000.hdf5', key='procdf', mode='w')
		procdf.to_hdf('../alpha/processed_out/processed_sourceRot33_ICPC_Pb_241Am_10000000.hdf5', key='procdf', mode='w')



def gauss_fit_func(x, A, mu, sigma, C):
	# return (A * (np.exp(-1.0 * ((x - mu)**2) / (2 * sigma**2))+C) +D)
	# return (A * (np.exp(-1.0 * ((x - mu)**2) / (2 * sigma**2))+C))
	return (A * (1/(sigma*np.sqrt(2*np.pi))) *(np.exp(-1.0 * ((x - mu)**2) / (2 * sigma**2))+C))
	# return (A * (np.exp(-1.0 * ((x - mu)**2) / (2 * sigma**2))))

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


	fig, ax = plt.subplots(ncols=3)
	nbins=50
	counts, xbins, ybins = np.histogram2d(x, y, bins=nbins, normed=True)
	ax[0].hist2d(x, y, bins=nbins, cmap='plasma', normed=True)
	# plt.scatter(x, y, c=energy, s=1, cmap='plasma')
	# cb = plt.colorbar()
	# cb.set_label("Energy (keV)", ha = 'right', va='center', rotation=270, fontsize=14)
	# cb.ax.tick_params(labelsize=12)
	ax[0].set_xlim(-10,10)
	ax[0].set_ylim(9,19)

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
	# print(xi.flatten())
	# exit()
	# ax[1].pcolormesh(xi, yi, zi.reshape(xi.shape), cmap='plasma')
	ax[1].pcolormesh(xi, yi, zi, cmap='plasma')
	# ax[1].pcolormesh(xi, yi, norm_zi.reshape(xi.shape), cmap='plasma')
	ax[1].set_xlim(-10,10)
	ax[1].set_ylim(9,19)

	levels = [0.1]

	# contour_hist = ax[2].contour(counts.T,extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],cmap='plasma')
	# CS = ax[2].contour(xi, yi, zi.reshape(xi.shape), cmap='plasma')
	CS = ax[2].contour(xi, yi, zi, cmap='plasma')
	# CSF = ax[2].contourf(xi, yi, norm_zi.reshape(xi.shape), cmap='plasma')
	# CSF = ax[2].contourf(xi, yi, zi.reshape(xi.shape), cmap='plasma')
	# plt.clabel(CS, fmt = '%2.1d', colors = 'k', fontsize=14)
	ax[2].clabel(CS, fmt = '%.2f', fontsize=20)
	CB = plt.colorbar(CS, shrink=0.8, extend='both')

	ax[2].set_xlim(-10,10)
	ax[2].set_ylim(9,19)
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

def plotSpot(filename, source=False, particle = 'all'):

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


	fig, ax = plt.subplots()
	plt.scatter(x, y, c=energy, s=1, cmap='plasma', norm=LogNorm(1,6000))
	# plt.scatter(x, y, c=energy, s=1, cmap='plasma')
	cb = plt.colorbar()
	cb.set_label("Energy (keV)", ha = 'right', va='center', rotation=270, fontsize=14)
	cb.ax.tick_params(labelsize=12)
	plt.xlim(-40,40)
	plt.ylim(-40,40)
	ax.set_xlabel('x position (mm)', fontsize=16)
	ax.set_ylabel('y position (mm)', fontsize=16)
	plt.setp(ax.get_xticklabels(), fontsize=14)
	plt.setp(ax.get_yticklabels(), fontsize=14)
	plt.title(plot_title, fontsize=16)
	plt.show()

	if source==True:
		source_df = pd.read_hdf(filename, keys='sourcePV_df')
		sourceEnergy = np.array(source_df['energy']*1000)
		x_source = np.array(source_df['x'])
		print(len(x_source))

# def plot1DSpot((filename, source=False, axis = 1, particle = 'all', fit=True, plot=True))
#
# 	# Read datafile and create dataframes
# 	# df = pandarize(filename)
# 	df = pd.read_hdf(filename, keys='procdf')
# 	xfit_min = -2.5
# 	xfit_max = 2.5
# 	yfit_min = -20.
# 	yfit_max = 20.
#
# 	if particle == 'all':
# 		x = np.array(df['x'])
# 		y = np.array(df['y'])
# 		z = np.array(df['z'])
# 		energy = np.array(df['energy']*1000)
# 		plot_title = 'Spot Size, $^{241}$Am 10$^7$ Primaries, all energies'
#
# 	elif particle == 'alpha':
# 		alpha_df = df.loc[df.energy > 5]
# 		x = np.array(alpha_df['x'])
# 		y = np.array(alpha_df['y'])
# 		z = np.array(alpha_df['z'])
# 		energy = np.array(alpha_df['energy']*1000)
# 		plot_title = 'Spot Size, $^{241}$Am 10$^7$ Primaries, Energy $>$ 5 MeV'
#
# 	elif particle == 'gamma':
# 		gamma_df = df.loc[(df.energy > .04) & (df.energy < 0.08)]
# 		x = np.array(gamma_df['x'])
# 		y = np.array(gamma_df['y'])
# 		z = np.array(gamma_df['z'])
# 		energy = np.array(gamma_df['energy']*1000)
# 		plot_title = 'Spot Size, $^{241}$Am 10$^7$ Primaries, 60 kev $<$ Energy $<$ 80 keV'
#
# 	else:
# 		print('specify particle type!')
# 		exit()
#
# 	if axis==1:
# 		Print('Fitting x data points')
# 		xfit_df = df.loc[(df.x > xfit_min) & (df.x < xfit_max)]
# 		xfit = np.array(xfit_df['x'])
#
# 	elif axis==2:
# 		Print('Fitting y data points')
# 		xfit_df = df.loc[(df.y > yfit_min) & (df.y < yfit_max)]
# 		xfit = np.array(xfit_df['y'])
#
#
# 	# max = np.amax(xfit)
# 	# min = np.amin(xfit)
# 	# print('max: ', max, 'min: ', min)
# 	# exit()
#
# 	# array_mean = np.mean(xfit)
# 	array_stdv = np.std(xfit)
# 	# array_med = np.median(xfit)
# 	# print('mean = ', array_mean, ', median = ', array_med, ', stdev = ', array_stdv, ', fwhm = ' , 2.355*array_stdv)
# 	# exit()
# 	# print(xfit)
#
# 	fig, ax = plt.subplots(figsize=(8,6))
# 	# ax.figure(figsize=(10,10))
#
# 	# Bins for histogram
# 	# bins = np.linspace(-36, 36, 1440)
# 	bins = np.linspace(-10,10, 80)
# 	# bins = np.linspace((-5.)*array_stdv,(5.)*array_stdv, 100)
#
# 	print('bins =  ', len(bins))
#
# 	# Create histogram which will be fit to, without drawing the histogram
# 	xvals, xbins = np.histogram(xfit, bins=bins)
# 	fitbins = xbins[:-1] + np.diff(xbins) / 2 #xbins gives bin edges for the histogram. Want the fit to plot based on the center of the bins. This centers it.
# 	# pltbins = np.linspace(-5.*array_stdv,5*array_stdv,1000) #plot fit with finer x resolution independent of the binning of the histogram
# 	pltbins = np.linspace(-2.5,2.5,1000) #plot fit with finer x resolution independent of the binning of the histogram
# 	# print(xbins, xvals)
# 	# print('bins = ', len(xbins))
# 	# print('fitbins: ', len(fitbins))
# 	# exit()
#
# 	#Fit the histogram, get fit parameters and covariances
# 	popt, pcov = curve_fit(gauss_fit_func, xdata=fitbins, ydata=xvals, method = 'lm')
# 	perr = np.sqrt(np.abs(np.diag(pcov))) # Variances of each fit parameter
# 	print(perr)
# 	# print(pcov)
# 	# print('mean = ', popt[1], 'stdev = ', popt[2], 'C = ', popt[3])
# 	# print('mean = ', popt[1], 'stdev = ', popt[2])
# 	print('popt: ', popt)
# 	sigma = np.abs(popt[2])
# 	print('sigma ', sigma)
# 	sigma_uncertainty = np.sqrt(perr[2])
# 	sigma_percentUncertainty = (sigma_uncertainty/sigma)*100
# 	print(sigma_percentUncertainty)
# 	fwhm = sigma*2.355
# 	array_fwhm = array_stdv*2.355
# 	print('FWHM = ', fwhm,  '; array FWHM = ', array_fwhm)
#
# 	# Draw full histogram (including points not used in the fit), along with the fit result
# 	ax.hist(x, bins = bins, label = 'Data: %.f entries \n 0.25 mm bins' % len(x))
# 	ax.plot(pltbins, gauss_fit_func(pltbins, *popt), 'r-', linewidth = 2, label='Fit: FWHM = %.2f mm \n Entries for fit: %.f' % (fwhm, len(xfit)))
#
# 	plt.xlim(-10,10)
# 	# plt.ylim(-31,31)
#
# 	legend = ax.legend(loc='upper right', shadow = False, fontsize='12')
#
# 	ax.set_xlabel('x position (mm)', fontsize=14)
# 	plt.setp(ax.get_xticklabels(), fontsize=12)
# 	plt.setp(ax.get_yticklabels(), fontsize=12)
# 	plt.title('Spot Size, $^{241}$Am 10$^7$ Primaries, no cuts. FWHM: %.2f mm' % fwhm, fontsize=14)
#
# 	plt.show()

def plot1DSpot(filename, axis='x', particle = 'all', fit=True):
	axis = str(axis)
	particle = str(particle)
	df = pd.read_hdf(filename, keys='procdf')
	energy = np.array(df['energy'])

	if particle=='all':
		scale_std = 1.5
		if axis=='x':
			x = np.array(df['x'])
		elif axis=='y':
			x = np.array(df['y'])
		else:
			print('Specify fit axis! Can be x or y')
			exit()

	elif particle=='alpha':
		alpha_df = df.loc[df.energy > 5]
		scale_std = 2.
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
	print('median: ', median, ' std: ', std, ' mean: ', mean)
	q50 = np.quantile(x, 0.5)
	# q1 = np.quantile(x, 0.2)
	# q2 = np.quantile(x, 0.8)
	# q1 = np.quantile(x, 0.001)
	# q2 = np.quantile(x, 0.997)
	q1 = -30
	q2 = 30
	print(q1, q2)
	print(q2-q1)
	# exit()


	# minbin = int(round(median-(scale_std*std)))
	# maxbin = int(round(median+(scale_std*std)))

	minbin = int(round(q1))
	maxbin = int(round(q2))

	print('minbin: ', minbin, ' minvalue: ', minvalue)
	print('maxbin: ', maxbin, ' maxvalue: ', maxvalue)
	# exit()

	if int(round(amin)) > minbin:
		print('Previous minbin: ', minbin)
		minbin = minvalue
		print('New minbin: ', minbin)
	if int(round(amax)) < maxbin:
		maxbin = maxvalue
	bin_scale = 5
	binno = int(round(maxbin-minbin)*bin_scale)
	print('minbin: ', minbin, ' maxbin: ',  maxbin)
	# print(median+std/2)
	# exit()

	if axis=='x':
		xfit_df = df.loc[(df.x > q1) & (df.x < q2)]
		xfit = np.array(xfit_df['x'])
	elif axis=='y':
		xfit_df = df.loc[(df.y > q1) & (df.y < q2)]
		xfit = np.array(xfit_df['y'])

	# if axis=='x':
	# 	xfit_df = df.loc[(df.x > (median-(scale_std*std))) & (df.x < (median+(scale_std*std)))]
	# 	xfit = np.array(xfit_df['x'])
	# elif axis=='y':
	# 	xfit_df = df.loc[(df.y > (median-(scale_std*std))) & (df.y < (median+(scale_std*std)))]
	# 	xfit = np.array(xfit_df['y'])

	bins = np.linspace(minbin, maxbin, binno)
	# bins = np.linspace(0, 30, binno)
	std = np.std(xfit)

	fit_entries = len(xfit)



	# ax.hist(y, bins=bins)
	# plt.show()

	fig, ax = plt.subplots(figsize=(8,6))
	# ax.figure(figsize=(10,10))

	# Bins for histogram
	# bins = np.linspace(-36, 36, 1440)
	# bins = np.linspace(-10,10, 80)
	# bins = np.linspace((-5.)*array_stdv,(5.)*array_stdv, 100)

	print('bins =  ', len(bins))

	# Create histogram which will be fit to, without drawing the histogram
	# xvals, xbins = np.histogram(xfit, bins=bins)
	xvals, xbins = np.histogram(xfit, bins=bins)
	max_fitval = np.amax(xvals)
	print(max_fitval)
	# exit()
	fitbins = xbins[:-1] + np.diff(xbins) / 2 #xbins gives bin edges for the histogram. Want the fit to plot based on the center of the bins. This centers it.
	# print('xbins:', xbins ,'fitbins:', fitbins)
	# exit()
	# pltbins = np.linspace(-5.*array_stdv,5*array_stdv,1000) #plot fit with finer x resolution independent of the binning of the histogram
	pltbins = np.linspace(minbin,maxbin,1000) #plot fit with finer x resolution independent of the binning of the histogram
	# print(xbins, xvals)
	# print('bins = ', len(xbins))
	# print('fitbins: ', len(fitbins))
	# exit()

	#Fit the histogram, get fit parameters and covariances
	# popt, pcov = curve_fit(gauss_fit_func, xdata=fitbins, ydata=xvals, p0=[1, median, std, 1])
	# perr = np.sqrt(np.abs(np.diag(pcov))) # Variances of each fit parameter
	# print(perr)
	# print(pcov)
	# print(type(popt))
	# exit()

	# print('Amplitude: ', popt[0], ' Fit mean = ', popt[1], ' Fit Stdv = ', popt[2], 'C = ', popt[3])
	# print('mean = ', popt[1], 'stdev = ', popt[2])
	# print('popt: ', popt)

	# sigma = np.abs(popt[2])

	# fit_amp = popt[0]



	# print('sigma ', sigma)

	# sigma_uncertainty = np.sqrt(perr[2])

	# sigma_percentUncertainty = (sigma_uncertainty/sigma)*100

	# print(sigma_percentUncertainty)

	# fwhm = sigma*2.355
	# array_fwhm = std*2.355

	# print('FWHM = ', fwhm,  '; array FWHM = ', array_fwhm)

	# Draw full histogram (including points not used in the fit), along with the fit result
	plt_minbin = int(round(np.amin(x)))
	plt_maxbin = int(round(np.amax(x)))
	# plt_binno = len(x) * (binno/fit_entries)*(plt_maxbin-plt_minbin)
	plt_binno = int(np.abs((maxbin-minbin)/(plt_maxbin-plt_minbin))*(len(x)/len(xfit))*binno)
	print(plt_minbin, plt_maxbin, plt_binno)


	# exit()
	# x_pltbins = np.linspace(plt_minbin, plt_maxbin, 30)
	# x_pltbins = np.linspace(0, 30, binno)
	x_pltbins = np.linspace(minbin, maxbin, binno)

	plt_xvals, plt_xbins = np.histogram(x, x_pltbins)
	max_histval = np.amax(plt_xvals)
	scale_fit = (max_histval/max_fitval)
	# print('fit_amp: ', fit_amp, 'scaled_amp: ', scale_fit)
	# print('max_fitval: ', max_fitval, 'max_histval: ', max_histval, 'fit amp: ', fit_amp)




	ax.hist(x, bins = x_pltbins, label = 'Data: %.f entries \n 0.25 mm bins' % len(x))

	# ax.plot(pltbins, gauss_fit_func(pltbins, *popt), 'r-', linewidth = 2, label='Fit: FWHM = %.2f mm \n Entries for fit: %.f' % (fwhm, len(xfit)))
	# ax.plot(pltbins, gauss_fit_func(pltbins, scale_fit*popt[0], popt[1], popt[2], popt[3]), 'r-', linewidth = 2, label='Fit: FWHM = %.2f mm \n Entries for fit: %.f' % (fwhm, len(xfit)))



	# plt.xlim(-10,10)

	# plt.ylim(-31,31)



	legend = ax.legend(loc='upper right', shadow = False, fontsize='12')



	ax.set_xlabel('x position (mm)', fontsize=14)

	plt.setp(ax.get_xticklabels(), fontsize=12)

	plt.setp(ax.get_yticklabels(), fontsize=12)

	# plt.title('Spot Size, $^{241}$Am 10$^7$ Primaries, no cuts. FWHM: %.2f mm' % fwhm, fontsize=14)



	plt.show()



# def plot1DSpot(filename):
#
	# # Read datafile and create dataframes
	# # df = pandarize(filename)
	# df = pd.read_hdf(filename, keys='procdf')
	# energy = np.array(df['energy'])
	# alpha_df = df.loc[df.energy > 5]
	# gamma_df = df.loc[(df.energy > .04) & (df.energy < 0.08)]
	#
	# # Create numpy arrays from dataframe
	#
	# x = np.array(df['x'])
	# y = np.array(df['y'])
	# z = np.array(df['z'])
	# # x = np.array(alpha_df['x'])
	# # y = np.array(alpha_df['y'])
	# # z = np.array(alpha_df['z'])
	# # x = np.array(gamma_df['x'])
	# # y = np.array(gamma_df['y'])
	# # z = np.array(gamma_df['z'])
	#
	# # create new dataframe with cut on the x positions to fit ony between -2.5 and 2.5 mm
	# xfit_df = df.loc[(df.x > -2.5) & (df.x < 2.5)]
	# # xfit_df = alpha_df.loc[(alpha_df.x > -2.5) & (alpha_df.x < 2.5)]
	# # xfit_df = gamma_df.loc[(gamma_df.x > -2.5) & (gamma_df.x < 2.5)]
	# xfit = np.array(xfit_df['x'])
	# # max = np.amax(xfit)
	# # min = np.amin(xfit)
	# # print('max: ', max, 'min: ', min)
	# # exit()
	#
	# # array_mean = np.mean(xfit)
	# array_stdv = np.std(xfit)
	# # array_med = np.median(xfit)
	# # print('mean = ', array_mean, ', median = ', array_med, ', stdev = ', array_stdv, ', fwhm = ' , 2.355*array_stdv)
	# # exit()
	# # print(xfit)
	#
	# fig, ax = plt.subplots(figsize=(8,6))
	# # ax.figure(figsize=(10,10))
	#
	# # Bins for histogram
	# # bins = np.linspace(-36, 36, 1440)
	# bins = np.linspace(-10,10, 80)
	# # bins = np.linspace((-5.)*array_stdv,(5.)*array_stdv, 100)
	#
	# print('bins =  ', len(bins))
	#
	# # Create histogram which will be fit to, without drawing the histogram
	# xvals, xbins = np.histogram(xfit, bins=bins)
	# fitbins = xbins[:-1] + np.diff(xbins) / 2 #xbins gives bin edges for the histogram. Want the fit to plot based on the center of the bins. This centers it.
	# # pltbins = np.linspace(-5.*array_stdv,5*array_stdv,1000) #plot fit with finer x resolution independent of the binning of the histogram
	# pltbins = np.linspace(-2.5,2.5,1000) #plot fit with finer x resolution independent of the binning of the histogram
	# # print(xbins, xvals)
	# # print('bins = ', len(xbins))
	# # print('fitbins: ', len(fitbins))
	# # exit()
	#
	# #Fit the histogram, get fit parameters and covariances
	# popt, pcov = curve_fit(gauss_fit_func, xdata=fitbins, ydata=xvals, method = 'lm')
	# perr = np.sqrt(np.abs(np.diag(pcov))) # Variances of each fit parameter
	# print(perr)
	# # print(pcov)
	# # print('mean = ', popt[1], 'stdev = ', popt[2], 'C = ', popt[3])
	# # print('mean = ', popt[1], 'stdev = ', popt[2])
	# print('popt: ', popt)
	# sigma = np.abs(popt[2])
	# print('sigma ', sigma)
	# sigma_uncertainty = np.sqrt(perr[2])
	# sigma_percentUncertainty = (sigma_uncertainty/sigma)*100
	# print(sigma_percentUncertainty)
	# fwhm = sigma*2.355
	# array_fwhm = array_stdv*2.355
	# print('FWHM = ', fwhm,  '; array FWHM = ', array_fwhm)
	#
	# # Draw full histogram (including points not used in the fit), along with the fit result
	# ax.hist(x, bins = bins, label = 'Data: %.f entries \n 0.25 mm bins' % len(x))
	# ax.plot(pltbins, gauss_fit_func(pltbins, *popt), 'r-', linewidth = 2, label='Fit: FWHM = %.2f mm \n Entries for fit: %.f' % (fwhm, len(xfit)))
	#
	# plt.xlim(-10,10)
	# # plt.ylim(-31,31)
	#
	# legend = ax.legend(loc='upper right', shadow = False, fontsize='12')
	#
	# ax.set_xlabel('x position (mm)', fontsize=14)
	# plt.setp(ax.get_xticklabels(), fontsize=12)
	# plt.setp(ax.get_yticklabels(), fontsize=12)
	# plt.title('Spot Size, $^{241}$Am 10$^7$ Primaries, no cuts. FWHM: %.2f mm' % fwhm, fontsize=14)
	#
	# plt.show()


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
