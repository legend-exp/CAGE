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
    time_intervals = 60
    print(find_jumps(dg, plot))
    
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

def hist_jump_in_run(run, lh5_dir=lh5_dir):
    ehists = []
    blhists = []
    r = run['run'].iloc[0]
    cycles = run['cycle']
    for c in cycles:
        cyc = run.query(f'cycle == {c}')
        f_dsp = f"{lh5_dir}/dsp/{cyc['dsp_file'].values[0]}"
        try:
            dsp = h5py.File(f_dsp)
        except OSError:
            print('Cannot find file ', f_dsp)
            continue
        trapEftp = np.array(dsp['ORSIS3302DecoderForEnergy']['dsp']['trapEftp'])
        baseline = np.array(dsp['ORSIS3302DecoderForEnergy']['dsp']['bl'])
        ts = np.array(dsp['ORSIS3302DecoderForEnergy']['dsp']['timestamp'])
        ts_corrected = correct_timestamps(f_dsp)
        if ts_corrected[-1] < time_intervals:
            print(f'Cycle {c} is less than {time_intervals} seconds')
            continue
        bl = np.mean(baseline)
        adu1460 = find_1460(ts_corrected, trapEftp)
        blh = np.histogram2d(ts_corrected, baseline, bins=[np.arange(0, ts_corrected[-1], np.minimum(time_intervals, int(ts_corrected[-1]-1))), np.arange(bl - 100, bl+100)])
        eh = np.histogram2d(ts_corrected, trapEftp, bins=[np.arange(0, ts_corrected[-1], np.minimum(time_intervals, int(ts_corrected[-1]-1))), np.arange(adu1460-250, adu1460+250)])
        ehists.append((eh, c))
        blhists.append((blh, c))
    return ehists, blhists

# TODO: Use better thresholds than hardcoded ones
def find_jump_in_run(cycles, ehist_infos, bhist_infos, thresholds, plot):
    assert len(ehist_infos) == len(bhist_infos)
    ret = []
    for i in np.arange(len(ehist_infos)):
        c = ehist_infos[i][1]
        ehist, t_edges, e_edges = ehist_infos[i][0]
        bhist, t_edges, b_edges = bhist_infos[i][0]
        ye = [e_edges[np.argmax(ehist[j][:])] for j in range(0, len(t_edges)-1)]
        yemean = np.mean(ye)
        ye -= yemean
        yb = [b_edges[np.argmax(bhist[j][:])] for j in range(0, len(t_edges)-1)]
        ybmean = np.mean(yb)
        yb -= ybmean
        step = np.hstack((np.ones(int(len(ye))), -1*np.ones(int(len(ye)))))
        econvolved = np.convolve(ye, step, mode='valid')
        bconvolved = np.convolve(yb, step, mode='valid')
        estep_index = np.argmax(np.abs(econvolved)) 
        bstep_index = np.argmax(np.abs(bconvolved)) 
        eleft = np.mean(ye[np.maximum(estep_index-10, 0):np.maximum(estep_index-1,0)])
        eright = np.mean(ye[np.minimum(estep_index+1, len(ye)):np.minimum(estep_index+10,len(ye))])
        bleft = np.mean(yb[np.maximum(bstep_index-10, 0):np.maximum(bstep_index-1,0)])
        bright = np.mean(yb[np.minimum(bstep_index+1, len(yb)):np.minimum(bstep_index+10,len(yb))])
        ejump = eright-eleft
        bjump = bright-bleft     
        if np.abs(ejump) > thresholds[0] or np.abs(bjump) > thresholds[1]:
            if plot:
                os.makedirs(os.path.dirname("./plots/jumpdet/"), exist_ok=True)
                plt.figure()
                plt.imshow(np.transpose(ehist), extent=(t_edges[0], t_edges[-1], e_edges[0], e_edges[-1]), aspect='auto', origin='lower')
                plt.plot(t_edges[:-1], ye + yemean, color='red')
                plt.axvline(t_edges[estep_index],color='orange')
                plt.xlabel('timestamp')
                plt.ylabel('trapEftp')
                plt.title(f'cycle {c} trapEftp')
                plt.savefig(f'./plots/jumpdet/cycle{cycles.iloc[i]}_trapEftp.png', dpi=300)
                plt.figure()
                plt.imshow(np.transpose(bhist), extent=(t_edges[0], t_edges[-1], b_edges[0], b_edges[-1]), aspect='auto', origin='lower')
                plt.plot(t_edges[:-1], yb + ybmean, color='red')
                plt.axvline(t_edges[bstep_index],color='orange')
                plt.xlabel('timestamp')
                plt.ylabel('baseline')
                plt.title(f'cycle {c} baseline')
                plt.savefig(f'./plots/jumpdet/cycle{cycles.iloc[i]}_baseline.png', dpi=300)
            ret.append( (t_edges[estep_index], ejump, c, 'e') )
            ret.append( (t_edges[bstep_index], bjump, c, 'b') )
    return ret

#a jump is a list [run, cycle, time, adu]
def find_jumps(dg, plot):
    en_jumps = ['energy']
    bl_jumps = ['baseline']
    runs = dg.fileDB['run'].unique()
    for i in range(len(runs)):
        r = runs[i]
        run = dg.fileDB.query(f'run == {r}')
        ehists, blhists = hist_jump_in_run(run)
        jumps = find_jump_in_run(run['cycle'], ehists, blhists, [15, 3], plot)
        if jumps is not None:
            for jump in jumps:
                if jump[-1] == 'e':
                    ejump = [r, jump[2], jump[0], jump[1]]
                    en_jumps.append(ejump)
                if jump[-1] == 'b':
                    bjump = [r, jump[2], jump[0], jump[1]]
                    bl_jumps.append(bjump)
    return (en_jumps, bl_jumps)

if __name__=="__main__":
    main()