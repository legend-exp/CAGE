#!/usr/bin/env python3
import h5py
import numpy as np
import matplotlib.pyplot as plt

f_dsp = '/Users/wisecg/Data/LH5/cage/dsp/cage_run110_cyc1185_dsp.lh5'

dsp = h5py.File(f_dsp)
timestamp = np.array(dsp['ORSIS3302DecoderForEnergy']['dsp']['timestamp'])
timestamp = timestamp.astype(np.int64)

# rollover correction.
clock = 100e6 # 100 MHz
UINT_MAX = 4294967295 # (0xffffffff)
t_max = UINT_MAX / clock
#ts = df_hit['timestamp'].values / clock
ts = timestamp / clock
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

plt.plot(np.arange(len(ts_sec)), ts_sec)
plt.show()