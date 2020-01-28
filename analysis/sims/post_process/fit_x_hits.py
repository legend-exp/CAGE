import h5py
import pandas as pd
import sys
import numpy as np
from decimal import Decimal
import pygama.analysis.histograms as pgh
#from pygama.utils import plot_func
import pygama.utils as pgu
import pygama.analysis.peak_fitting as pga
import matplotlib.pyplot as plt
plt.style.use('style.mplstyle')

def main():

    fit_weighted_hits()

def fit_weighted_hits():

    if(len(sys.argv) != 2):
        print('Usage: x_hits.py [Source]')
        sys.exit()

    df =  pd.read_hdf("processed.hdf5", key="procdf")

    m = np.array(df['x'])

    def gauss(x, mu, sigma, A, C):
        """
        define a gaussian distribution, w/ args: mu, sigma, amplitude and constant background.
        """
        return A * np.exp(-(x - mu)**2 / (2. * sigma**2)) + C

    fit_range = (-3,3)

    hist, bins, var = pgh.get_hist(m, range=fit_range, dx=0.5)
    #pgh.plot_hist(hist, bins, var=None, label="data")
    pars, cov = pga.fit_hist(gauss, hist, bins, var=var, guess=[0,2,10000,20])
    pgu.print_fit_results(pars, cov, gauss)
    pgu.plot_func(gauss, pars, range=fit_range, label="chi2 fit", color='red')
    plt.hist(m, np.arange(-32,32,0.5), histtype='step', color='black')

    FWHM = '%.2f' % abs(Decimal(pars[1]*2))

    plt.ylim(0,plt.ylim()[1])
    plt.xlim(-31,31)
    plt.xlabel('x position (mm)', ha='right', x=1.0)
    plt.ylabel('Counts', ha='right', y=1.0)
    plt.title('Fitting Weighted Position Measurements ('+sys.argv[1]+' Source), FWHM = '+str(FWHM)+' mm')
    plt.legend(frameon=True, loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
        main()

