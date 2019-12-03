#!/usr/bin/env python3
import os
import argparse
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('clint.mpl')
from pprint import pprint
import scipy.signal as signal
import itertools

from pygama import DataSet
import pygama.utils as pu
import pygama.analysis.histograms as ph
import pygama.analysis.peak_fitting as pf


def main():
    """
    """
    
    par = argparse.ArgumentParser(description="pygama dsp optimizer")
    arg, st, sf = par.add_argument, "store_true", "store_false"
    arg("-ds", nargs='*', action="store", help="load runs for a DS")
    arg("-r", "--run", nargs=1, help="load a single run")
    args = vars(par.parse_args())

    # -- standard method to declare the DataSet from cmd line --
    ds = pu.get_dataset_from_cmdline(args, "runDB.json", "calDB.json")
    
    
    
    # set I/O locations
    f_grid = "~/Data/cage/cage_optimize_resuts.h5"

    if ["grid"]:
        # set the combination of processor parameters to vary to optimize resolution
        set_grid(f_grid)
    
    if["window"]:
        # generate a small single-peak file w/ uncalibrated energy
        window_ds(ds)
    
    # create an hdf5 file with DataFrames for each set of parameters
    # process_ds(f_grid)
    
    # fit all outputs to the peakshape function and find the best resolution
    # get_fwhm(f_grid, test=False)
    
    # show results
    plot_fwhm()
    
    
def set_grid():
    """
    to get the best energy resolution, we want to explore the possible values
    of our DSP processor list, especially trap filter and RC decay constants.
    
    a flexible + easy way to vary a bunch of parameters at once is to create
    a DataFrame with each row corresponding to a set of parameters.  
    We then use this DF as an input/output for the other functions.
    
    it could also easily be extended to loop over detector channels, or vary
    any other set of parameters in the processor list ......
    """
    # # this is pretty ambitious, but maybe doable -- 3500 entries
    # e_rises = np.arange(1, 6, 0.2)
    # e_flats = np.arange(0.5, 4, 0.5)
    # rc_consts = np.arange(50, 150, 5) # ~same as MJD charge trapping correction
    
    # this runs more quickly -- 100 entries, 3 minutes on my mac
    e_rises = np.arange(2, 3, 0.2)
    e_flats = np.arange(1, 3, 1)
    rc_consts = np.arange(52, 152, 10)
    
    lists = [e_rises, e_flats, rc_consts]
    
    prod = list(itertools.product(*lists)) # clint <3 stackoverflow
    
    df = pd.DataFrame(prod, columns=['rise','flat','rc']) 
    
    # print(df)
    
    return df

    
def window_ds():
    """
    Take a single DataSet and window it so that the file only contains events 
    near an expected peak location.
    Create some temporary in/out files s/t the originals aren't overwritten.
    """
    # run = 42
    # ds = DataSet(run=run, md="runDB.json")
    ds_num = 3
    ds = DataSet(ds_num, md="runDB.json")
    
    # specify temporary I/O locations
    p_tmp = "~/Data/cage"
    f_tier1 = "~/Data/cage/cage_ds3_t1.h5"
    f_tier2 = "~/Data/cage/cage_ds3_t2.h5"
    
    # figure out the uncalibrated energy range of the K40 peak
    # xlo, xhi, xpb = 0, 2e6, 2000 # show phys. spectrum (top feature is 2615 pk)
    xlo, xhi, xpb = 990000, 1030000, 250 # k40 peak, ds 3

    t2df = ds.get_t2df()
    hE, xE = ph.get_hist(t2df["energy"], range=(xlo, xhi), dx=xpb)
    plt.semilogy(xE, hE, ls='steps', lw=1, c='r')
    
    import matplotlib.ticker as ticker
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.4e'))
    plt.locator_params(axis='x', nbins=5)

    plt.xlabel("Energy (uncal.)", ha='right', x=1)
    plt.ylabel("Counts", ha='right', y=1)
    plt.savefig(f"./plots/cage_ds{ds_num}_winK40.pdf")
    # exit()
        
    # write a windowed tier 1 file containing only waveforms near the peak
    t1df = pd.DataFrame()
    for run in ds.paths:
        ft1 = ds.paths[run]["t1_path"]
        print(f"Scanning ds {ds_num}, run {run}\n    file: {ft1}")
        for chunk in pd.read_hdf(ft1, 'ORSIS3302DecoderForEnergy', chunksize=5e4):
            t1df_win = chunk.loc[(chunk.energy > xlo) & (chunk.energy < xhi)]
            print(t1df_win.shape)
            t1df = pd.concat([t1df, t1df_win], ignore_index=True)
    
    # -- save to HDF5 output file -- 
    h5_opts = {
        "mode":"w", # overwrite existing
        "append":False, 
        "format":"table",
        "complib":"blosc:zlib",
        "complevel":1,
        "data_columns":["ievt"]
        }
    t1df.reset_index(inplace=True)
    t1df.to_hdf(f_tier1, key="df_windowed", **h5_opts)
    print("wrote file:", f_tier1)


def process_ds(df_grid):
    """
    and determine the trapezoid parameters that minimize
    the FWHM of the peak (fitting to the peakshape function).
    
    NOTE: I don't think we need to multiprocess this, since that's already
    being done in ProcessTier1
    """
    from pygama.dsp.base import Intercom
    from pygama.io.tier1 import ProcessTier1
    import pygama.io.decoders.digitizers as pgd
    
    ds_num = 3
    ds = DataSet(ds_num, md="runDB.json")
    first_run = ds.runs[0]
    
    # specify temporary I/O locations
    out_dir = os.path.expanduser('~') + "/Data/cage"
    t1_file = f"{out_dir}/cage_ds3_t1.h5"
    t2_file = f"{out_dir}/cage_ds3_t2.h5"
    opt_file = f"{out_dir}/cage_ds3_optimize.h5"
    
    if os.path.exists(opt_file):
        os.remove(opt_file)
        
    # check the windowed file
    # tmp = pd.read_hdf(t1_file)
    # nevt = len(tmp)

    t_start = time.time()
    for i, row in df_grid.iterrows():
        
        # estimate remaining time in scan
        if i == 4:
            diff = time.time() - t_start
            tot = diff/5 * len(df_grid) / 60
            tot -= diff / 60
            print(f"Estimated remaining time: {tot:.2f} mins")
        
        rise, flat, rc = row
        print(f"Row {i}/{len(df_grid)},  rise {rise}  flat {flat}  rc {rc}")
        
        # custom tier 1 processor list -- very minimal
        proc_list = {
            "clk" : 100e6,
            "fit_bl" : {"ihi":500, "order":1},
            "blsub" : {},
            "trap" : [
                {"wfout":"wf_etrap", "wfin":"wf_blsub", 
                 "rise":rise, "flat":flat, "decay":rc},
                {"wfout":"wf_atrap", "wfin":"wf_blsub", 
                 "rise":0.04, "flat":0.1, "fall":2} # could vary these too
                ],
            "get_max" : [{"wfin":"wf_etrap"}, {"wfin":"wf_atrap"}],
            # "ftp" : {"test":1}
            "ftp" : {}
        }
        proc = Intercom(proc_list)
        
        dig = pgd.SIS3302Decoder
        dig.decoder_name = "df_windowed"
        dig.class_name = None
        
        # process silently
        ProcessTier1(t1_file, proc, output_dir=out_dir, overwrite=True, 
                     verbose=False, multiprocess=True, nevt=np.inf, ioff=0, 
                     chunk=ds.runDB["chunksize"], run=first_run, 
                     t2_file=t2_file, digitizers=[dig])
        
        # load the temporary file and append to the main output file
        df_key = f"opt_{i}"
        t2df = pd.read_hdf(t2_file)
        t2df.to_hdf(opt_file, df_key)


def get_fwhm(df_grid, test=False):
    """
    duplicate the plot from Figure 2.7 of Kris Vorren's thesis (and much more!)
    
    this code fits the e_ftp peak to the HPGe peakshape function (same as in
    calibration.py) and writes a new column to df_grid, "fwhm".
    """
    out_dir = "~/Data/cage"
    opt_file = f"{out_dir}/cage_ds3_optimize.h5"
    print("input file:", opt_file)
    
    # declare a new column for resolution in df_grid
    df_grid["fwhm"] = np.nan
    
    # loop over the keys and fit each e_ftp spectrum to the peakshape function
    for i, row in df_grid.iterrows():
        
        key = f"opt_{i}"
        rise, flat, rc = row
        
        t2df = pd.read_hdf(opt_file, key=f"opt_{i}")
        
        # histogram spectrum near the uncalibrated peak -- have to be careful here
        xlo, xhi, xpb = 2550, 2660, 1
        hE, xE, vE = ph.get_hist(t2df["e_ftp"], range=(xlo, xhi), dx=xpb, trim=False)
        
        # set initial guesses for the peakshape function.  most are pretty rough
        mu = xE[np.argmax(hE)]
        sigma = 5
        hstep = 0.001
        htail = 0.5
        tau = 10
        bg0 = np.mean(hE[:20])
        amp = np.sum(hE)
        x0 = [mu, sigma, hstep, htail, tau, bg0, amp]
        
        xF, xF_cov = pf.fit_hist(pf.radford_peak, hE, xE, var=vE, guess=x0)

        # update the master dataframe with the resolution value
        df_grid.at[i, "fwhm"] = xF[1] * 2.355

        if test:
            # plot every dang fit 
            print(row)
            
            plt.cla()

            # peakshape function
            plt.plot(xE, pf.radford_peak(xE, *x0), c='orange', label='guess')
            plt.plot(xE, pf.radford_peak(xE, *xF), c='r', label='peakshape')
            
            plt.axvline(mu, c='g')
            
            # plot individual components
            # tail_hi, gaus, bg, step, tail_lo = pf.radford_peak(xE, *xF, components=True)
            # gaus = np.array(gaus)
            # step = np.array(step)
            # tail_lo = np.array(tail_lo)
            # plt.plot(xE, gaus * tail_hi, ls="--", lw=2, c='g', label="gaus+hi_tail")
            # plt.plot(xE, step + bg, ls='--', lw=2, c='m', label='step + bg')
            # plt.plot(xE, tail_lo, ls='--', lw=2, c='k', label='tail_lo')
        
            plt.plot(xE[1:], hE, ls='steps', lw=1, c='b', label="data")
            plt.plot(np.nan, np.nan, c='w', label=f"fwhm = {results[key][0]:.2f} uncal.")
            plt.plot(np.nan, np.nan, c='w', label=f"rt = {rt:.2f} us")
        
            plt.xlabel("Energy (uncal.)", ha='right', x=1)
            plt.ylabel("Counts", ha='right', y=1)
            plt.legend(loc=2)
            
            plt.show()
            
            
def plot_fwhm():
    """
    """
    # make the FWHM^2 vs risetime plot
    # pprint(results)
    
    fwhms = [results[key][0]**2 for key in results]
    rts = [results[key][1] for key in results]
    
    plt.plot(rts, fwhms, ".", c='b')
    
    plt.xlabel("Ramp time (us)", ha='right', x=1)
    plt.ylabel(r"FWHM$^2$", ha='right', y=1)
    
    # plt.show()
    plt.savefig("./plots/cage_ds3_fwhm2.pdf")
            
            

    
if __name__=="__main__":
    main()