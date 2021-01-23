#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('../clint.mpl')

from pygama import DataGroup
import pygama.io.lh5 as lh5
import pygama.analysis.histograms as pgh


def main():
    """
    OPPI biasing on 12/30/20 didn't go well (elog 298).  
    It looks like there is significantly more capacitance "near" the detector,
    and during running, I thought I saw it changing over time (~minute timescale).
    
    Study: find some pulse shape parameters that are sensitive to "how slow" 
    the waveforms really are, for a given run/cycle.  We should then be able
    to extend this analysis to monitor the stability of OPPI during "good" runs.
    """
    # fileDB query.  this could be moved to cmd line arg in the future
    que = 'run==111' # 30 min bad bkg run
    # que = 'run==110' # 3hr alpha run
    
    # load fileDB.  use DataGroup, with a hack to fix use of relative file paths
    pwd = os.getcwd()
    os.chdir('../processing/')
    dg = DataGroup('cage.json', load=True)
    os.chdir(pwd)
    # print(dg.fileDB)
    
    # run query
    dg.fileDB = dg.fileDB.query(que)
    
    # check query result
    view_cols = ['daq_file', 'cycle', 'run', 'runtype', 'startTime', 'runtime']
    print(dg.fileDB[view_cols], '\n')
    
    # -- run routines -- 
    plot_dsp(dg)
    # show_wfs(dg)
    
    
def plot_dsp(dg):
    """
    create a DataFrame from the dsp files and make some 1d and 2d diagnostic plots.
    
    for reference, current 12/30/20 dsp parameters:
      ['channel', 'timestamp', 'energy', 'bl', 'bl_sig', 'trapEftp',
       'trapEmax', 'triE', 'tp_max', 'tp_0', 'tp_10', 'tp_50', 'tp_80',
       'tp_90', 'A_10', 'AoE', 'dcr_raw', 'dcr_max', 'dcr_ftp', 'hf_max']
    columns added by this code:
      ['run', 'cycle', 'ts_sec', 'ts_glo']
    """
    sto = lh5.Store()
    
    dsp_name = 'ORSIS3302DecoderForEnergy/dsp'
    wfs_name = 'ORSIS3302DecoderForEnergy/raw/waveform'
    
    def get_dsp_dfs(df_row):
        """
        grab the dsp df, add some columns, and return it
        """
        f_dsp = dg.lh5_dir + '/' + df_row.dsp_path + '/' + df_row.dsp_file
        if len(f_dsp) > 1:
            print('Error, this part is supposed to only load individual files')
            exit()
        f_dsp = f_dsp.iloc[0]
        run, cyc = df_row.run.iloc[0], df_row.cycle.iloc[0]
        # print(run, cyc, f_dsp)
        
        # grab the dataframe and add some columns
        tb, nr = sto.read_object(dsp_name, f_dsp)
        df = tb.get_dataframe()
        df['run'] = run
        df['cycle'] = cyc
        
        # need global timestamp.  just calculate here instead of making hit files
        clock = 100e6 # 100 MHz
        UINT_MAX = 4294967295 # (0xffffffff)
        t_max = UINT_MAX / clock
        ts = df['timestamp'].values / clock
        tdiff = np.diff(ts)
        tdiff = np.insert(tdiff, 0 , 0)
        iwrap = np.where(tdiff < 0)
        iloop = np.append(iwrap[0], len(ts))
        ts_new, t_roll = [], 0
        for i, idx in enumerate(iloop):
            ilo = 0 if i==0 else iwrap[0][i-1]
            ihi = idx
            ts_block = ts[ilo:ihi]
            t_last = ts[ilo-1]
            t_diff = t_max - t_last
            ts_new.append(ts_block + t_roll)
            t_roll += t_last + t_diff
        df['ts_sec'] = np.concatenate(ts_new)
        t_start = df_row.startTime.iloc[0]
        df['ts_glo'] = df['ts_sec'] + t_start
        
        # print(df)
        return df
        
    # create the multi-cycle DataFrame
    df_dsp = dg.fileDB.groupby(['cycle']).apply(get_dsp_dfs)
    df_dsp.reset_index(inplace=True, drop=True) # << VERY IMPORTANT!
    
    print(df_dsp)
    print(df_dsp.columns)
    
    # 1. 1d energy histogram -- use this to select energy range of interest
    et = 'trapEmax'
    elo, ehi, epb = 0, 10000, 10
    edata = df_dsp.trapEmax.values
    hist, bins, _ = pgh.get_hist(edata, range=(elo, ehi), dx=epb)
    plt.semilogy(bins[1:], hist, ds='steps', c='b', lw=1)
    plt.xlabel(et, ha='right', x=1)
    plt.ylabel('Counts', ha='right', y=1)
    # plt.show()
    plt.savefig('./plots/risingedge_1dspec.pdf')
    plt.cla()
    
    # 2. 2d histo: show risetime vs. time for wfs in an energy range
    
    # choose risetime range (usec)
    # rlo, rhi, rpb = 0, 5, 0.1 # run 110 (good)
    rlo, rhi, rpb = 0, 50, 1 # run 111 (bad)

    # select energy range
    elo, ehi, epb = 1500, 1600, 0.5
    df = df_dsp.query(f'trapEmax > {elo} and trapEmax < {ehi}').copy()

    # calculate timestamp range
    t0 = df_dsp.iloc[0]['ts_glo']
    df['ts_adj'] = (df.ts_glo - t0) / 60 # minutes after t0
    tlo, thi, tpb = 0, df.ts_adj.max(), 1
    
    # compute t50-100 risetime
    df['rt_us'] = (df.tp_max - df.tp_50) / 1e3 # convert ns to us
    # print(df[['tp_max', 'tp_50', 'rt_us']])
    
    nbx, nby = int((thi-tlo)/tpb), int((rhi-rlo)/rpb)
    plt.hist2d(df['ts_adj'], df['rt_us'], bins=[nbx, nby],
               range=[[tlo, thi], [rlo, rhi]], cmap='jet')
    
    plt.xlabel('Time (min)', ha='right', x=1)
    plt.ylabel('Rise Time (t50-100), usec', ha='right', y=1)
    # plt.show()
    plt.savefig('./plots/risingedge_2dRisetime.png', dpi=150)
    plt.cla()
    
    
    # 3. 1st 10 wfs from energy region selection (requires raw file)
    # this assumes the first file has 10 events
    db = dg.fileDB.iloc[0]
    cyc = db.cycle
    f_raw = dg.lh5_dir + '/' + db.raw_path + '/' + db.raw_file
    f_dsp = dg.lh5_dir + '/' + db.dsp_path + '/' + db.dsp_file
    
    edata = lh5.load_nda([f_dsp], ['trapEmax'], dsp_name)['trapEmax']
    idx = np.where((edata >= elo) & (edata <= ehi))
    
    nwfs = 10
    idx_sel = idx[0][:nwfs]
    n_rows = idx_sel[-1] + 1 # read up to this event and stop
    tb_wfs, n_wfs = sto.read_object(wfs_name, f_raw, n_rows=n_rows)
    
    # grab the 2d numpy array of waveforms
    wfs = tb_wfs['values'].nda[idx_sel, :]

    ts = np.arange(0, len(wfs[0,:-2])) / 1e2 # usec
    
    for iwf in range(wfs.shape[0]):
        plt.plot(ts, wfs[iwf,:-2], lw=2, alpha=0.5)
    
    plt.xlabel('Time (us)', ha='right', x=1)
    plt.ylabel('ADC', ha='right', y=1)
    
    plt.show()
    
    
    
if __name__=='__main__':
    main()