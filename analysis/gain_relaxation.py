import h5py
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from pygama.analysis.peak_fitting import gauss_mode_width_max
from pygama import DataGroup, lh5


def main():
    doc="""
    Detect jumps in the 1460 keV line in the uncalibrated trapEftp
    """
    rthf = argparse.RawTextHelpFormatter
    par = argparse.ArgumentParser(description=doc, formatter_class=rthf)
    arg, st, sf = par.add_argument, 'store_true', 'store_false'
    arg('-q', '--query', nargs=1, type=str,
        help="select file group to calibrate: -q 'run==1' ")
    arg('-p', '--plot', action='store_true',
        help="plot option")
    arg('-u', '--user', action='store_true',
        help="use lh5 user directory")
    args = par.parse_args()
    plot = args.plot
    user = args.user

    # load main DataGroup, select files from cmd line
    dg = DataGroup('$CAGE_SW/processing/cage.json', load=True)
    if args.query:
        que = args.query[0]
        dg.fileDB.query(que + "and skip==False", inplace=True)
    else:
        dg.fileDB = dg.fileDB[-1:]
    
    global lh5_dir
    if user:
        lh5_dir = os.path.expandvars(dg.lh5_user_dir)
    else:
        lh5_dir = os.path.expandvars(dg.lh5_dir)
        
    global time_intervals
    time_intervals = 300
    global fit_interval
    fit_interval = 1800
    print(find_drifts(dg, False))
    
    return 0

def correct_timestamps(f_dsp):
    dsp = h5py.File(f_dsp)
    ts_old = np.array(dsp['ORSIS3302DecoderForEnergy']['dsp']['timestamp'])
    ts_old = ts_old.astype(np.int64)
    ts_sec = []
    clock = 100e6 # 100 MHz
    UINT_MAX = 4294967295 # (0xffffffff)
    t_max = UINT_MAX / clock
    ts = ts_old/ clock
    tdiff = np.diff(ts)
    tdiff = np.insert(tdiff, 0 , 0)
    iwrap = np.where(tdiff < 0)
    iloop = np.append(iwrap[0], len(ts))
    ts_new, t_roll = [], 0
    for i, idx in enumerate(iloop):
        ilo = 0 if i==0 else iwrap[0][i-1]
        ihi = idx
        ts_block = ts[ilo:ihi]
        ts_block = (np.array(ts_block)).astype(np.uint64)
        ts_new.append(ts_block + t_roll)
        t_last = ts[ilo-1]
        t_diff = t_max - t_last
        t_roll += t_last + t_diff
    ts_sec = np.concatenate(ts_new)
    return ts_sec

def find_1460(timestamps, trapEftp):
    ehist, t_edges, e_edges = np.histogram2d(timestamps,  trapEftp, bins=[np.arange(0, timestamps[-1], np.minimum(time_intervals, int(timestamps[-1]-1))), np.arange(1000,5000, 10)])
    ind = np.unravel_index(np.argmax(ehist), ehist.shape)
    return e_edges[ind[1]]

def hist_1460_in_run(dg, plot=False):    
    cycles = dg.fileDB['cycle'].unique()

    f_dsp = f"{lh5_dir}/dsp/{df['dsp_file'].iloc[-1]}"
    dsp = h5py.File(f_dsp)

    ts_corrected = correct_timestamps(f_dsp)
    trapEftp = np.array(dsp['ORSIS3302DecoderForEnergy']['dsp']['trapEftp'])
    baseline = np.array(dsp['ORSIS3302DecoderForEnergy']['dsp']['bl'])

    adu1460 = find_1460(ts_corrected, trapEftp)
    bl = np.mean(baseline)

    energy_bins = np.arange(adu1460-100, adu1460+101)
    bl_bins = np.arange(bl-50, bl+51)
    time_bins = np.arange(0, df['startTime'].iloc[-1] - df['startTime'].iloc[0] + ts_corrected[-1], time_intervals)
    time_bins = np.append(time_bins, df['startTime'].iloc[-1] - df['startTime'].iloc[0] + ts_corrected[-1])


    ehists = np.zeros((len(time_bins)-1, len(energy_bins)-1))
    blhists = np.zeros((len(time_bins)-1, len(bl_bins)-1))


    for i in range(len(df)):
        f_dsp = f"{lh5_dir}/dsp/{df['dsp_file'].iloc[i]}"
        try:
            dsp = h5py.File(f_dsp)
        except OSError:
            continue
        trapEftp = np.array(dsp['ORSIS3302DecoderForEnergy']['dsp']['trapEftp'])
        baseline = np.array(dsp['ORSIS3302DecoderForEnergy']['dsp']['bl'])


        ts_corrected = np.array(correct_timestamps(f_dsp) + df['startTime'].iloc[i] - df['startTime'].iloc[0])

        e_cyc = np.histogram2d(ts_corrected, trapEftp, bins=[time_bins, energy_bins])
        b_cyc = np.histogram2d(ts_corrected, baseline, bins=[time_bins, bl_bins])
        
        ehists += e_cyc[0]
        blhists += b_cyc[0]

    if plot:
        e_data = list(zip([(t, e) for t in time_bins[:-1] for e in energy_bins[:-1]]))
        e_data = [e[0] for e in e_data]
        b_data = list(zip([(t, b) for t in time_bins[:-1] for b in bl_bins[:-1]]))
        b_data = [b[0] for b in b_data]

        ex = [e[0] for e in e_data]
        ey = [e[1] for e in e_data]
        bx = [b[0] for b in b_data]
        by = [b[1] for b in b_data]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24,10))
        fig.suptitle(f'Run {run}, Cycles {cycles.iloc[0]} - {cycles.iloc[-1]}')

        ax1.hist2d(ex, ey, bins=[time_bins, energy_bins], weights=np.ravel(ehists))
        ax1.set(xlabel='Timestamp (s)', ylabel='trapEftp (adu)') 

        ax2.hist2d(bx, by, bins=[time_bins, bl_bins], weights=np.ravel(blhists))
        ax2.set(xlabel='Timestamp (s)', ylabel='Baseline (adu)')
    return ehists, blhists, time_bins, energy_bins, bl_bins

#dates should be a string in the format YYYY-MM-DDTHH:MM in UTC (will maybe support timezones later)
#e.g. 2021-06-26T15:00
def hist_1460_over_time(dg, start_date, end_date, plot=True):
    time_intervals = 900 
    dt_start = dt.fromisoformat(start_date)
    dt_start = dt_start.replace(tzinfo=timezone.utc)
    dt_end = dt.fromisoformat(end_date)
    dt_end = dt_end.replace(tzinfo=timezone.utc)

    
    time_start = dt.timestamp(dt_start)
    time_end = dt.timestamp(dt_end)
    
    df = dg.fileDB.query(f'startTime >= {time_start} & startTime <= {time_end}')
    f_dsp = f"{lh5_dir}/dsp/{df['dsp_file'].iloc[0]}"
    dsp = h5py.File(f_dsp)
    ts_corrected = correct_timestamps(f_dsp)
    trapEftp = np.array(dsp['ORSIS3302DecoderForEnergy']['dsp']['trapEftp'])
    
    adu1460 = find_1460(ts_corrected, trapEftp)
    energy_bins = np.arange(adu1460-200, adu1460+201)
    time_bins = np.arange(time_start, time_end, time_intervals)

    ehists = np.zeros((len(time_bins)-1, len(energy_bins)-1))
    
    for i in range(len(df)):
        f_dsp = f"{lh5_dir}/dsp/{df['dsp_file'].iloc[i]}"
        try:
            dsp = h5py.File(f_dsp)
        except OSError:
            continue
        trapEftp = np.array(dsp['ORSIS3302DecoderForEnergy']['dsp']['trapEftp'])

        ts_corrected = np.array(correct_timestamps(f_dsp) + df['startTime'].iloc[i])

        e_cyc = np.histogram2d(ts_corrected, trapEftp, bins=[time_bins, energy_bins])
    
        e_cyc_norm = e_cyc[0]/np.maximum(np.amax(e_cyc[0]), 1)
        ehists += e_cyc_norm

    if plot:
        e_data = list(zip([(t, e) for t in time_bins[:-1] for e in energy_bins[:-1]]))
        e_data = [e[0] for e in e_data]

        ex = [e[0] for e in e_data]
        ey = [e[1] for e in e_data]

        fig, ax = plt.subplots(1,1,figsize=(12,10))
        fig.suptitle(f'1460 line from {start_date} to {end_date}')
                
        #print(np.amin(ehists))
        #print(np.amax(ehists))
        
        #dt_bins = [dt.fromtimestamp(time_bins[j], tz=timezone.utc) for j in range(len(time_bins))]
        #date_bins = [dt.isoformat(dt_bins[j]) for j in range(len(dt_bins))]

        hist = ax.hist2d(ex, ey, bins=[time_bins, energy_bins], weights=np.ravel(ehists), vmax=1)
        ax.set(xlabel='Time (UTC)', ylabel='trapEftp (adu)') 
        
        start_label = dt_start
        if start_label.time().hour < 12:
            start_label = start_label.replace(hour=12, minute=0)
        else:
            start_label = start_label.replace(day=(dt_start.date().day+1), hour=0, minute=0)
            
        start_label = start_label.replace(tzinfo=None)
        xlabels = [start_label + timedelta(hours=24*j) for j in range(int((time_end-time_start)/(24*3600)))]
        xticks = [dt.timestamp(xlabels[j]) for j in range(len(xlabels))]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, rotation=45, ha='right')
        
        fig.colorbar(hist[3])
        
        #ticks = ax.get_xticks()
        #new_labels = [dt.isoformat((dt.fromtimestamp(ticks[j], tz=timezone.utc)).replace(tzinfo=None), timespec='minutes') for j in range(len(ticks))]
        #ax.set_xticklabels(new_labels, rotation=45, ha="right")
 
    return ehists

def fit_peaks(ehists, blhists, time_bins, energy_bins, bl_bins, plot=False):
    e_total = []
    b_total = []
    for j in range(len(time_bins)-1):      
        e_max = np.amax(ehists[j][:])
        e_max_ind = np.argmax(ehists[j][:])
        
        if e_max < 5:
            e_total.append((0,0))
            b_total.append((0,0))
            continue
        
        e_where = np.where(ehists[j][:] >= e_max)[0]
        #print(e_where)
        e_fwhm = np.maximum(np.abs(energy_bins[e_max_ind]-energy_bins[e_where[0]]), np.abs(energy_bins[e_max_ind]-energy_bins[e_where[-1]]))
        
                
        e_total.append((energy_bins[e_max_ind], e_fwhm))
        
        b_max = np.amax(blhists[j][:])
        b_max_ind = np.argmax(blhists[j][:])
    
        b_where = np.where(blhists[j][:] >= b_max)[0]
        b_fwhm = np.maximum(np.abs(bl_bins[b_max_ind]-bl_bins[b_where[0]]), np.abs(bl_bins[b_max_ind]-bl_bins[b_where[-1]]))
                
        b_total.append((bl_bins[b_max_ind], b_fwhm))
                

        if plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24,10))
            fig.suptitle(f'Time {time_bins[j]}-{time_bins[j+1]}')

            ax1.hist(energy_bins[:-1], bins=energy_bins, weights=ehists[j])
            ax1.set(xlabel='trapEftp (adu)', ylabel='count')

            ax2.hist(bl_bins[:-1], bins=bl_bins, weights=blhists[j])
            ax2.set(xlabel='baseline (adu)', ylabel='count')            


    return e_total, b_total

# Returns: slope of up to first hour of data in run
def find_slope_in_run(time_bins, e_total, b_total):
    x = time_bins[-1]
    if x > fit_interval: 
        x = fit_interval
    if x < time_intervals:
        x = 2
    else:
        x /= time_intervals
    
    try:
        #print(time_bins[:int(x)])
        e_fit = np.polyfit(time_bins[:int(x)], [e_total[j][0] for j in range(int(x))], 1, w=1/np.sqrt([e_total[j][1] + 1 for j in range(int(x))]), cov=True)
        b_fit = np.polyfit(time_bins[:int(x)], [b_total[j][0] for j in range(int(x))], 1, w=1/np.sqrt([b_total[j][1] + 1 for j in range(int(x))]), cov=True)
    except np.linalg.LinAlgError as err:
        print(err)
        return None
    return e_fit, b_fit
    
    
#a drift is a list [run, e_fit, b_fit]
def find_drifts(dg, plot=False):
    drifts = []
    
    runs = dg.fileDB['run'].unique()      
    for i in range(len(runs)):
        r = runs[i]
        df = dg.fileDB.query(f'run == {r} and skip==False')
        ehists, blhists, time_bins, energy_bins, bl_bins = hist_1460_in_run(dg, r, False)
        e_total, b_total = fit_peaks(ehists, blhists, time_bins, energy_bins, bl_bins)

        x = time_bins[-1]
        if x > fit_interval: 
            x = fit_interval
        if x < time_intervals:
            print(f'Run {r} is shorter than {time_intervals} seconds')
            continue
        else:
            x /= time_intervals
        
        fit = find_slope_in_run(time_bins, e_total, b_total)
        
        if fit is not None:
            efit = fit[0]
            bfit = fit[1]
            efit_slope = efit[0][0]
            efit_int = efit[0][1]
            efit_unc = efit[1][0][0]
            bfit_slope = bfit[0][0]
            bfit_int = bfit[0][1]
            bfit_unc = bfit[1][0][0]
            
            #print(np.abs(efit_slope)*x*time_intervals, np.std([e_total[j][0] for j in range(len(e_total))]))
            #print(np.abs(bfit_slope)*x*time_intervals, np.std([b_total[j][0] for j in range(len(b_total))]))

            #print(np.std([e_total[j][0] for j in range(int(len(e_total)/3), int(len(e_total)*2/3))]))
            delta_e = np.abs(efit_slope)*x*time_intervals
            delta_b = np.abs(bfit_slope)*x*time_intervals
            sigma_e = np.std([e_total[j][0] for j in range(int(len(e_total)/3), len(e_total))])
            sigma_b = np.std([b_total[j][0] for j in range(int(len(b_total)/3), len(b_total))])
            
            
            
            if (delta_e > 5*sigma_e and np.abs(efit_slope) > 1e-3) or (delta_b > 5*sigma_b and np.abs(bfit_slope) > 1e-3):            
                #print("1460: ", delta_e, 6*sigma_e)
                #print("Baseline: ", delta_b, 6*sigma_b)
                
                #if sigma_e == 0 or sigma_b == 0:
                #    print([e_total[j][0] for j in range(int(len(e_total)/3), len(e_total))])
                #    print([b_total[j][0] for j in range(int(len(b_total)/3), len(b_total))])

                drifts.append((r,(efit_slope, efit_int, efit_unc), (bfit_slope, bfit_int, bfit_unc)))
        
            if plot:
                x = time_bins[-1]
                if x > fit_interval: 
                    x = fit_interval
                if x < time_intervals:
                    x = 2
                else:
                    x /= time_intervals

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24,10))
                fig.suptitle(f'Run {r}')

                #ax1.errorbar(time_bins[:-1], [e[0] for e in e_total], xerr=None, yerr=[e[1] for e in e_total], ecolor='r' )
                ax1.plot(time_bins[:-1], [e[0] for e in e_total])
                ax1.plot(time_bins[:int(x)], efit_slope*time_bins[:int(x)] + efit_int, label='slope = {:.2E} +- {:.2E}'.format(efit_slope, efit_unc), color='orange', linewidth=4)
                ax1.set(xlabel='timestamps', ylabel='trapEftp (adc)', ylim=(np.max(e_total[:][0])-50, (np.max(e_total[:][0])+50)))
                ax1.legend()

                #ax2.errorbar(time_bins[:-1],  [b[0] for b in b_total], xerr=None, yerr=[b[1] for b in b_total], ecolor='r')
                ax2.plot(time_bins[:-1],  [b[0] for b in b_total])
                ax2.plot(time_bins[:int(x)], bfit_slope*time_bins[:int(x)] + bfit_int, label='slope = {:.2E} +- {:.2E}'.format(bfit_slope, bfit_unc), color='orange', linewidth=4)
                ax2.set(xlabel='timestamps', ylabel='baseline (adc)', ylim=(np.max(b_total[:][0])-30, (np.max(b_total[:][0])+30)))
                ax2.legend()
                plt.show()
        elif plot:
            x = time_bins[-1]
            if x > fit_interval: 
                x = fit_interval
            x /= time_intervals

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24,10))
            fig.suptitle(f'Run {r}')

            ax1.plot(time_bins[:-1], [e[0] for e in e_total])
            ax1.set(xlabel='timestamps', ylabel='trapEftp (adc)', ylim=(np.max(e_total[:][0])-50, (np.max(e_total[:][0])+50)))

            ax2.plot(time_bins[:-1],  [b[0] for b in b_total])
    
            ax2.set(xlabel='timestamps', ylabel='baseline (adc)', ylim=(np.max(b_total[:][0])-30, (np.max(b_total[:][0])+30)))
                
    return drifts
if __name__=="__main__":
    main()