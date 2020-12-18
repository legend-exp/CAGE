#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from io import StringIO
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
plt.style.use('../clint.mpl')

from pygama import DataGroup
import pygama.io.lh5 as lh5
import pygama.analysis.histograms as pgh
import pygama.analysis.peak_fitting as pgf

def main():
    """
    """
    # Set 1 (Oct 2020)
    # https://elog.legend-exp.org/UWScanner/249 (one file per run)
    tb1 = StringIO("""
    V_pulser  run  E_keV  mV_firststage
    3.786 824  1460  300
    5.2   826  2000  420
    6.77  828  2615  544
    8.27  830  3180  660
    10    832  3840  800
    2.5   834  967   200
    1.3   836  500   106
    0.55  837  212   44
    0.16  838  60    13.2
    """)
    dfp1 = pd.read_csv(tb1, delim_whitespace=True)

    # Set 2 (Dec 2020)
    # https://elog.legend-exp.org/UWScanner/294
    tb2 = StringIO("""
    V_pulser  run  E_keV  mV_firststage
    0     1170  0     0
    3.76  1172  1460  316
    0.05  1173  15    7.2
    0.1   1174  31    11.0
    0.2   1175  62    19.4
    0.5   1176  167   44.0
    0.8   1177  277   69.6
    1     1178  352   85.6
    2     1179  744   172
    5     1180  1971  500
    8     1181  3225  740
    10    1182  4054  880
    """)
    dfp2 = pd.read_csv(tb2, delim_whitespace=True)

    # load fileDB to get dsp filenames
    # use DataGroup, with a temp hack to fix use of relative file paths
    pwd = os.getcwd()
    os.chdir('../processing/')
    dg = DataGroup('cage.json', load=True)
    os.chdir(pwd)

    # merge with fileDB
    cycles = dfp2['run'].tolist()
    df_pulsDB = dg.file_keys.loc[dg.file_keys['cycle'].isin(cycles)]
    df_pulsDB.reset_index(inplace=True)
    dfp2 = pd.concat([dfp2, df_pulsDB], axis=1)

    # -- run routines --
    # show_gain(dfp1, dfp2)
    # show_spectra(dfp2, dg)
    get_superpulses(dfp2, dg)


def show_gain(dfp1, dfp2):
    """
    Compare gain measurements (input V to output V from first stage).
    Also compare input V to output E.  Note these E values are from
    a rough ORCA calibration, and may be extra bad for Set 2 because
    of nonlinearity issues (see elog 294).
    """
    dfp1 = dfp1.sort_values('V_pulser')
    dfp2 = dfp2.sort_values('V_pulser')
    print(dfp1,'\n',dfp2)

    # 1. 1st stage output vs pulser input voltage
    plt.plot(dfp1.V_pulser, dfp1.mV_firststage, '.-r', ms=10, lw=1,
             label='Set 1 -- Oct. 2020')

    plt.plot(dfp2.V_pulser, dfp2.mV_firststage, '.-g', ms=10, lw=1,
             label='Set 2 -- Dec. 2020')

    plt.xlabel('V_pulser', ha='right', x=1)
    plt.ylabel('mV_firststage', ha='right', y=1)
    plt.legend()
    # plt.show()
    plt.savefig('./plots/pulser_firststage.pdf')
    plt.cla()

    # 2. Rough ORCA energy (keV) vs pulser input voltage
    plt.plot(dfp1.V_pulser, dfp1.E_keV, '.-r', ms=10, lw=1,
             label='Set 1 -- Oct. 2020')

    plt.plot(dfp2.V_pulser, dfp2.E_keV, '.-g', ms=10, lw=1,
             label='Set 2 -- Dec. 2020')

    plt.xlabel('V_pulser', ha='right', x=1)
    plt.ylabel('E (keV, rough cal)', ha='right', y=1)
    plt.legend()
    # plt.show()
    plt.savefig('./plots/pulser_energyrough.pdf')


def show_spectra(dfp, dg):
    """
    plot events from each pulser peak on top of a background spectrum run,
    to show where in the spectrum we sampled from.
    let's use the E_keV column to find the pulser peaks.
    need to figure out the proper calibration constant (use onboard energy)
    so load the bkg run and figure out the calibration constant.
    that's the parameter we need for get_superpulses.
    """
    run_diagnostic = False

    f_dsp = dg.lh5_dir + '/' + dfp.dsp_path + '/' + dfp.dsp_file
    f_bkg = f_dsp.iloc[0] # bkg run is 0 by dfn
    print('Background run:', f_bkg)

    # dataframe method - pulls all values from table
    # sto = lh5.Store()
    # tb_data, n_rows = sto.read_object('ORSIS3302DecoderForEnergy/dsp', f_bkg)
    # df_data = tb_data.get_dataframe()

    # load_nda method - just grab onboard energy
    tb_name = 'ORSIS3302DecoderForEnergy/dsp'
    edata = lh5.load_nda([f_bkg], ['energy'], tb_name)['energy']

    # use this flag to figure out the calibration of the 1460 line
    if run_diagnostic:
        elo, ehi, epb = 0, 1e7, 10000
        hist, bins, _ = pgh.get_hist(edata, range=(elo, ehi), dx=epb)
        plt.semilogy(bins[1:], hist, ds='steps', c='b', lw=1)
        plt.show()
        exit()

    ecal = 1460.8 / 2.005e6 # works for pulser dataset 2 (dec 2020)

    elo, ehi, epb = 0, 5000, 10
    hist, bins, _ = pgh.get_hist(edata * ecal, range=(elo, ehi), dx=epb)
    runtime = dfp.iloc[0].runtime * 60 # sec
    hist_rt = np.divide(hist, runtime)
    print(f'bkg runtime: {runtime:.2f} min')

    cmap = plt.cm.get_cmap('jet', len(dfp))
    for i, df_row in dfp.iterrows():

        epk, rt, vp = df_row[['E_keV', 'runtime', 'V_pulser']]
        rt *= 60 # sec
        if epk == 0: continue # skip the bkg run

        # draw the expected peak location based on our input table
        plt.axvline(epk, lw=1, alpha=0.5)

        # load pulser data
        f_cyc = dg.lh5_dir + '/' + df_row.dsp_path + '/' + df_row.dsp_file
        pdata = lh5.load_nda([f_cyc], ['energy'], tb_name)['energy'] * ecal

        # take a wide window around where we expect the pulser peak
        pdata = pdata[(pdata > epk-50) & (pdata < epk+50)]
        hp, bp, _ = pgh.get_hist(pdata, range=(elo, ehi), dx=epb)
        hp_rt = np.divide(hp, rt)
        plt.semilogy(bp[1:], hp_rt, ds='steps', lw=1, c=cmap(i),
                     label=f'{vp:.2f} V')

    plt.semilogy(bins[1:], hist_rt, ds='steps', c='k', lw=1,
                 label='bkg data')

    plt.xlabel(f'onboard energy (keV, c={ecal:.2e})', ha='right', x=1)
    plt.ylabel('cts / s', ha='right', y=1)
    plt.legend(fontsize=10)
    plt.show()
    # plt.savefig('./plots/transferfn_peaks.pdf')


def get_superpulses(dfp, dg):
    """
    calculate average waveforms for each set of pulser data
    """
    # find this with show_spectra
    ecal = 1460.8 / 2.005e6 # works for pulser dataset 2 (dec 2020)

    tb_name = 'ORSIS3302DecoderForEnergy/dsp'

    for i, df_row in dfp.iterrows():

        epk, rt, vp = df_row[['E_keV', 'runtime', 'V_pulser']]
        rt *= 60 # sec
        if epk == 0: continue # skip the bkg run

        # load pulser energies
        f_cyc = dg.lh5_dir + '/' + df_row.dsp_path + '/' + df_row.dsp_file

        pdata = lh5.load_nda([f_cyc], ['energy'], tb_name)['energy'] * ecal

        # take a more narrow window around the max pulser peak
        elo, ehi, epb = epk-50, epk+50, 0.5
        pdata_all = pdata[(pdata > elo) & (pdata < ehi)]
        hp, bp, _ = pgh.get_hist(pdata_all, range=(elo, ehi), dx=epb)
        pctr = bp[np.argmax(hp)]

        plo, phi, ppb = pctr-10, pctr+10, 0.1
        pdata = pdata[(pdata > plo) & (pdata < phi)]
        hp, bp, _ = pgh.get_hist(pdata_all, range=(plo, phi), dx=ppb)
        hp_rt = np.divide(hp, rt)
        hp_var = np.array([np.sqrt(h / (rt)) for h in hp])

        # fit a gaussian to get 1 sigma e-values
        ibin_bkg = 50
        bkg0 = np.mean(hp_rt[:ibin_bkg])
        b, h = bp[1:], hp_rt
        imax = np.argmax(h)
        upr_half = b[np.where((b > b[imax]) & (h <= np.amax(h)/2))][0]
        bot_half = b[np.where((b < b[imax]) & (h <= np.amax(h)/2))][-1]
        fwhm = upr_half - bot_half
        sig0 = fwhm / 2.355
        amp0 = np.amax(hp_rt) * fwhm
        p_init = [amp0, bp[imax], sig0, bkg0] # a, mu, sigma, bkg
        p_fit, p_cov = pgf.fit_hist(pgf.gauss_bkg, hp_rt, bp,
                                    var=hp_var, guess=p_init)
        fit_func = pgf.gauss_bkg
        xfit = np.arange(plo, phi, ppb * 0.1)
        plt.plot(xfit, fit_func(xfit, *p_init), '-', c='orange',
                 label='init')
        plt.plot(xfit, fit_func(xfit, *p_fit), '-', c='red',
                 label='fit')

        # plot data
        plt.plot(bp[1:], hp_rt, ds='steps', c='k', lw=1, label=f'{vp:.2f} V')

        plt.xlabel(f'onboard energy (keV, c={ecal:.2e})', ha='right', x=1)
        plt.ylabel('cts / s', ha='right', y=1)
        plt.legend(fontsize=10)
        plt.show()

        # now I want to select events within 1 sigma of the maximum
        # and pull all the waveforms to make a superpulse.
        print('lol tbd')


if __name__=="__main__":
    main()
