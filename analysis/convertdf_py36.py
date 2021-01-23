#!/usr/bin/env python3
"""
Jan 2021:
pandas hdf5 files generated with python3.8 have a different "pickle protocol"
than earlier versions of python (3.6 used by MJD container, Dec 2020), and can't
be read. This script just reads in an input file and makes a downgraded copy.

Run this script in the LEGEND image:
  $ shifter --image legendexp/legend-base:latest bash

This converts 'f_in' to a format readable by the current MJD image.
Once Dave updates the MJD container, we shouldn't need this script anymore.

Reference: https://stackoverflow.com/questions/63329657/python-3-7-error-unsupported-pickle-protocol-5
"""

import pickle
pickle.HIGHEST_PROTOCOL = 4
import pandas as pd

# f_in = './data/superpulses_dec2020.h5'
f_in = '../processing/fileDB.h5'

print('downgrading pickle protocol used in file:', f_in)

# loop over all dataframes stored in the file
with pd.HDFStore(f_in) as pf: 
    df_keys = pf.keys()
    print(df_keys)

for dfk in df_keys:
    df = pd.read_hdf(f_in, key=dfk)
    df.to_hdf(f_in, key=dfk)
    

