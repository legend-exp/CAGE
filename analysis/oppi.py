#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('clint.mpl')
from pprint import pprint

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
    """
    tier2_spec()
    tier1_wfs()
    
    
def tier2_spec():
    """
    show a few examples of energy spectra (onboard E and offline E)
    """
    run = 42
    ds = DataSet(run=run, md="runDB.json")
    t2df = ds.get_t2df()
    # print(t2df.columns)
    
    # # onboard E
    # ene = "energy"
    # # xlo, xhi, xpb = 0, 20e6, 5000 # show muon peak (full dyn. range)
    # xlo, xhi, xpb = 0, 2e6, 2000 # show phys. spectrum (top feature is 2615 pk)
    
    # # trap_max E
    # ene = "etrap_max"
    # xlo, xhi, xpb = 0, 50000, 100 # muon peak
    # xlo, xhi, xpb = 0, 6000, 10 # gamma spectrum
    
    # fixed time pickoff E 
    ene = "e_ftp"
    # xlo, xhi, xpb = 0, 50000, 100 # muon peak
    xlo, xhi, xpb = 0, 6000, 10 # gamma spectrum

    # get histogram
    hE, xE = ph.get_hist(t2df[ene], range=(xlo, xhi), dx=xpb)
    
    # make the plot
    plt.semilogy(xE, hE, ls='steps', lw=1, c='r', label=f'run {run}')
    plt.xlabel("Energy (uncal.)", ha='right', x=1)
    plt.ylabel("Counts", ha='right', y=1)
    
    # show a couple formatting tricks
    import matplotlib.ticker as ticker
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
    plt.grid(linestyle=':')
    
    plt.legend()
    # plt.show()
    plt.savefig(f"./plots/cage_run{run}_{ene}.pdf")
    

def tier1_wfs():
    """
    show some waveforms, with an example of a data cleaning cut.
    """
    run = 42
    ds = DataSet(run=run, md="runDB.json")
    ft1 = ds.paths[run]["t1_path"]
    t1df = pd.read_hdf(ft1, "ORSIS3302DecoderForEnergy", where="ievt < 10000")
    t1df.reset_index(inplace=True) # required step -- fix pygama "append" bug
    
    # for convenience, separate data columns from waveform columns
    d_cols, wf_cols = [], []
    for col in t1df.columns:
        if isinstance(col, int):
            wf_cols.append(col)
        else:
            d_cols.append(col)
    
    # get waveform dataframe
    wfs = t1df[wf_cols]
    
    # apply a cut
    idx_cut = t1df.index[t1df.energy > 10000].tolist()
    
    exit()

    
    # iterate over the waveform block and optionally save some waveform plots
    iwf = -1
    while True:
        if iwf != -1:
            inp = input()
            if inp == "q": exit()
            if inp == "p": iwf -= 2
        iwf += 1
        print(iwf)

        wf = wfs.iloc[iwf]
        ts = np.arange(len(wf))

        plt.cla()
        
        # raw waveform
        plt.plot(ts, wf, "-b", alpha=0.6, label='raw wf')
        
        # savitzky-golay smoothed
        wfsg = signal.savgol_filter(wf, window=47, order=2)
        plt.plot(ts, wfsg, "-r", label='savitzky-golay filter')

        plt.xlabel("clock ticks", ha='right', x=1)
        plt.ylabel("ADC", ha='right', y=1)
        plt.legend()
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.01)
    
    

def tier2():
    """
    """
    ds = DataSet(3, config="runDB.json")
    df = ds.get_t2df()
    print(df.columns)
    # df.hist(etype)
    # plt.show()
    
    


if __name__=="__main__":
    main()
