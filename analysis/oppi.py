#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('clint.mpl')
from pprint import pprint
import scipy.signal as signal

from pygama import DataSet
import pygama.analysis.histograms as ph

def main():
    """
    OPPI characterization suite

    pre:
    - put processors in test mode and tweak processor list
    need to tweak:  trap integration + flat top parameters, measure RC constant

    make a function for each one:
    - load t2 file and make energy spectrum
    - run calibration.py on run1
    - measure width of K40 peak (curve_fit: gaussian + linear BG)
    - look at A/E distrbution
    - run calibration.py (separately) and plot the calibration constants
    
    current tier 2 columns:
    ['channel', 'energy', 'energy_first', 'ievt', 'packet_id', 'timestamp',
     'ts_hi', 'bl_rms', 'bl_p0', 'bl_p1', 'dADC', 'fS', 'etrap_max',
     'etrap_imax', 'strap_max', 'strap_imax', 'atrap_max', 'atrap_imax',
     'ttrap_max', 'ttrap_imax', 'savgol_max', 'savgol_imax', 'current_max',
     'current_imax', 'tp5', 'tp10', 'tp50', 'tp90', 'tp100', 'n_curr_pks',
     's_curr_pks', 't0', 't_ftp', 'e_ftp', 'overflow', 'tslope_savgol',
     'tslope_pz', 'tail_amp', 'tail_tau']
    """
    # tier2_spec()
    # tier2_AoverE()
    tier1_wfs()
    
    
def tier2_spec():
    """
    show a few examples of energy spectra (onboard E and offline E)
    """
    run = 42
    ds = DataSet(run=run, md="runDB.json")
    t2df = ds.get_t2df()
    # print(t2df.columns)
    
    # onboard E
    ene = "energy"
    # xlo, xhi, xpb = 0, 20e6, 5000 # show muon peak (full dyn. range)
    xlo, xhi, xpb = 0, 2e6, 2000 # show phys. spectrum (top feature is 2615 pk)
    
    # # trap_max E
    # ene = "etrap_max"
    # xlo, xhi, xpb = 0, 50000, 100 # muon peak
    # xlo, xhi, xpb = 0, 6000, 10 # gamma spectrum
    
    # # fixed time pickoff E 
    # ene = "e_ftp"
    # # xlo, xhi, xpb = 0, 50000, 100 # muon peak
    # xlo, xhi, xpb = 0, 6000, 10 # gamma spectrum

    # get histogram
    hE, xE = ph.get_hist(t2df[ene], range=(xlo, xhi), dx=xpb)
    
    # make the plot
    plt.semilogy(xE, hE, ls='steps', lw=1, c='r', label=f'run {run}')
    plt.xlabel("Energy (uncal.)", ha='right', x=1)
    plt.ylabel("Counts", ha='right', y=1)
    
    # show a couple formatting tricks
    import matplotlib.ticker as ticker
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1e'))
    plt.locator_params(axis='x', nbins=5)
    plt.grid(linestyle=':')
    
    plt.legend()
    # plt.show()
    plt.savefig(f"./plots/cage_run{run}_{ene}.pdf")
    
    
def tier2_AoverE():
    """
    show the A/E distribution.
    """
    run = 42
    ds = DataSet(run=run, md="runDB.json")
    t2df = ds.get_t2df()
    
    # # 1d
    # xlo, xhi, xpb = -2000, 2000, 10
    # h, x = ph.get_hist(aoe, range=(xlo, xhi), dx=xpb)
    # plt.semilogy(x, h, ls='steps', lw=1, c='r', label=f'run {run}')
    # plt.xlabel("A/E (uncal.)", ha='right', x=1)
    # plt.ylabel("Counts", ha='right', y=1)
    # plt.grid(linestyle=':')
    # plt.legend()
    # # plt.show()
    # plt.cla()
    
    # 2d vs E
    aoe = t2df["atrap_max"] / t2df["e_ftp"] # not sure which of these is better, but 
    # aoe = t2df["current_max"] / t2df["e_ftp"] # this one has a weird 'sharp' feature
    
    xlo, xhi, xpb = 0, 6000, 5
    ylo, yhi, ypb = 0.6, 1.2, 0.001
    # ylo, yhi, ypb = 0, 0.1, 0.001
    nbx, nby = int((xhi-xlo)/xpb), int((yhi-ylo)/ypb)

    from matplotlib.colors import LogNorm
    plt.hist2d(t2df["e_ftp"], aoe, 
               bins=(nbx,nby), range=((xlo,xhi),(ylo,yhi)), 
               norm=LogNorm(), cmap='jet')

    # cb = plt.colorbar()
    # cb.set_label("Counts", ha='right', y=1)
    plt.xlabel("e_ftp (uncal.)", ha='right', x=1)
    plt.ylabel("A/E", ha='right', y=1)
    plt.grid(which='both', linestyle=':')

    plt.savefig(f"./plots/cage_run{run}_AE.png", dpi=200)


def tier1_wfs():
    """
    show some waveforms, with an example of a data cleaning cut.
    """
    run = 42
    iwf_max = 100000 # tier 1 files can be a lot to load into memory
    ds = DataSet(run=run, md="runDB.json")
    ft1 = ds.paths[run]["t1_path"]
    t1df = pd.read_hdf(ft1, "ORSIS3302DecoderForEnergy", where=f"ievt < {iwf_max}")
    t1df.reset_index(inplace=True) # required step -- fix pygama "append" bug
    
    # get waveform dataframe
    wf_cols = []
    for col in t1df.columns:
        if isinstance(col, int):
            wf_cols.append(col)
    wfs = t1df[wf_cols]
    
    # apply a cut based on the t1 columns
    # idx = t1df.index[(t1df.energy > 1.5e6)&(t1df.energy < 2e6)]
    
    # apply a cut based on the t2 columns
    ft2 = ds.paths[run]['t2_path']
    t2df = pd.read_hdf(ft2, where=f"ievt < {iwf_max}")
    t2df.reset_index(inplace=True)

    # t2df['AoE'] = t2df.current_max / t2df.e_ftp # scipy method
    t2df['AoE'] = t2df.atrap_max / t2df.e_ftp # trapezoid method
    
    idx = t2df.index[(t2df.AoE < 0.7)
                     &(t2df.e_ftp > 1000) & (t2df.e_ftp < 10000)
                     &(t2df.index < iwf_max)]

    wfs = wfs.loc[idx]
    wf_idxs = wfs.index.values # kinda like a TEntryList
    
    # make sure the cut output makes sense
    cols = ['ievt', 'timestamp', 'energy', 'e_ftp', 'atrap_max', 'current_max', 't0', 
            't_ftp', 'AoE', 'tslope_pz', 'tail_tau']
    print(t2df.loc[idx][cols].head())
    print(t1df.loc[idx].head())
    print(wfs.head())
    
    # iterate over the waveform block 
    iwf = -1
    while True:
        if iwf != -1:
            inp = input()
            if inp == "q": exit()
            if inp == "p": iwf -= 2
        iwf += 1
        iwf_cut = wf_idxs[iwf]
        
        # get waveform and dsp values
        wf = wfs.iloc[iwf]
        dsp = t2df.iloc[iwf_cut]
        ene = dsp.e_ftp
        aoe = dsp.AoE
        ts = np.arange(len(wf))
        
        # nice horizontal print of a pd.Series
        # print(iwf, iwf_cut)
        # print(wf.to_frame().T)
        # print(t2df.iloc[iwf_cut][cols].to_frame().T)
        
        plt.cla()
        plt.plot(ts, wf, "-b", alpha=0.9, label=f'e: {ene:.1f}, a/e: {aoe:.1f}')
        
        # savitzky-golay smoothed
        # wfsg = signal.savgol_filter(wf, 47, 2)
        wfsg = signal.savgol_filter(wf, 47, 1)
        plt.plot(ts, wfsg, "-r", label='savitzky-golay filter')

        plt.xlabel("clock ticks", ha='right', x=1)
        plt.ylabel("ADC", ha='right', y=1)
        plt.legend()
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.01)
    
    


if __name__=="__main__":
    main()
