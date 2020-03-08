import sys, h5py
import pandas as pd
import numpy as np

# Modify this value for different energy resolution
pctResAt1MeV = 0.15;

if(len(sys.argv) != 3):
    print('Usage: postprochdf5.py [inputfilename.hdf5] [outputname.hdf5]')
    sys.exit()

# have to open the input file with h5py (g4 doesn't write pandas-ready hdf5)
g4sfile = h5py.File(sys.argv[1], 'r')
g4sntuple = g4sfile['default_ntuples']['g4sntuple']
# print(g4sfile.keys())
# exit()


# pd.read_hdf('../geCounterOut.hdf5')


# build a pandas DataFrame from the hdf5 datasets we will use
g4sdf = pd.DataFrame(np.array(g4sntuple['event']['pages']), columns=['event'])
g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['step']['pages']), columns=['step']),
                   lsuffix = '_caller', rsuffix = '_other')
g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['Edep']['pages']), columns=['Edep']),
                   lsuffix = '_caller', rsuffix = '_other')
g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['volID']['pages']),
                   columns=['volID']), lsuffix = '_caller', rsuffix = '_other')
g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['iRep']['pages']),
                   columns=['iRep']), lsuffix = '_caller', rsuffix = '_other')
g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['pid']['pages']),
                   columns=['pid']), lsuffix = '_caller', rsuffix = '_other')
g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['x']['pages']),
                   columns=['x']), lsuffix = '_caller', rsuffix = '_other')
g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['y']['pages']),
                   columns=['y']), lsuffix = '_caller', rsuffix = '_other')
g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['z']['pages']),
                   columns=['z']), lsuffix = '_caller', rsuffix = '_other')

# print(np.array(g4sntuple['event']['pages']))
# exit()
# apply E cut / detID cut and sum Edeps for each event using loc, groupby, and sum
# write directly into output dataframe
detector_hits = g4sdf.loc[(g4sdf.Edep>0)&(g4sdf.volID==1)]
# print(detector_hits)
# exit()
procdf = pd.DataFrame(detector_hits.groupby(['event','volID'], as_index=False)['Edep'].sum())
procdf = procdf.rename(columns={'Edep':'energy'})

# apply energy resolution function
procdf['energy'] = procdf['energy'] + np.sqrt(procdf['energy'])*pctResAt1MeV/100.*np.random.randn(len(procdf['energy']))

# write to output file
procdf.to_hdf(sys.argv[2], key='procdf', mode='w')
print('wrote output')
