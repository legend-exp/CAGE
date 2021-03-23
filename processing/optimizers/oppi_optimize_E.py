import sys, json
import numpy as np
from collections import OrderedDict
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from pygama.dsp.dsp_optimize import *
from pygama import lh5
import pygama.analysis.histograms as pgh
from energy_selector import select_energies
import os


if len(sys.argv) < 2:
    print("Usage: oppi_optimize_E [lh5 raw file(s)]")
    sys.exit()

# get input file, dsp_config, and apdb
filenames = sys.argv[1:]
dsp_id = '01'
with open(os.path.expandvars(f'$CAGE_SW/processing/metadata/dsp/dsp_{dsp_id}.json')) as f: dsp_config = json.load(f, object_pairs_hook=OrderedDict)
with open('oppi_apdb.json') as f: apdb = json.load(f, object_pairs_hook=OrderedDict)

# Override dsp_config['outputs'] to contain only what we need for optimization
dsp_config['outputs'] = ['trapEftp', 'ct_corr']

# build a parameter grid for the dsp of choice
trap_grid = ParGrid()

ramp_values = np.linspace(1., 4., 5)
ramp_values = [ f'{ramp:.2f}*us' for ramp in ramp_values]
ftp_values = [ f'tp_0+({ramp}+2.5*us)' for ramp in ramp_values]
trap_grid.add_dimension('wf_trap', 1, ramp_values, companions=[('trapEftp', 1, ftp_values)])

# set up the figure-of-merit to be computed at each grid point
def ct_corr_E_var(tb_out, verbosity):
    EE = tb_out['trapEftp'].nda
    cc = tb_out['ct_corr'].nda / EE

    # bad gretina waveforms: need to cull them
    # first remove crazy ct_corr values
    idx = np.where((cc > 0) & (cc < 1) & (EE > 0))
    EE = EE[idx]
    cc = cc[idx]

    # now zoom in on energy twice
    E_ave = np.average(EE)
    idx = np.where((EE > 0.9*E_ave) & (EE < 1.1*E_ave))
    EE = EE[idx]
    cc = cc[idx]
    E_ave = np.average(EE)
    idx = np.where((EE > 0.99*E_ave) & (EE < 1.01*E_ave))
    EE = EE[idx]
    cc = cc[idx]

    # now go to +/- 3*sigma
    # it's non-gaus so this should be wide enough
    E_ave = np.average(EE)
    E_3sig = 3.*np.sqrt(np.var(EE))
    idx = np.where((EE > E_ave-E_3sig) & (EE < E_ave+E_3sig))
    EE = EE[idx]
    cc = cc[idx]

    # do a PCA for a first guess
    E_ave = np.average(EE)
    c_ave = np.average(cc)
    pca = PCA(n_components=2)
    pca.fit(np.vstack((EE-E_ave, cc-c_ave)).T)
    i_max = np.argmax(pca.explained_variance_)
    dE, dc = pca.components_[i_max]
    EEc = EE - dE/dc*cc

    # now cut in EEc
    Ec_ave = np.average(EEc)
    Ec_3sig = 3.*np.sqrt(np.var(EEc))
    idx = np.where((EEc > Ec_ave-Ec_3sig) & (EEc < Ec_ave+Ec_3sig))
    EE = EE[idx]
    cc = cc[idx]

    # now move to histograms, and vary dE until the peak is sharpest
    dEs = np.linspace(0, 2, 21)*dE
    bins = 100
    hrange = (Ec_ave - 3*Ec_3sig, Ec_ave + 3*Ec_3sig)
    max_height = 0
    max_dE = 0
    for dE_i in dEs:
        hist, bins, var = pgh.get_hist(EE - dE_i/dc*cc, bins, hrange)
        height = np.amax(hist)
        if height > max_height:
            max_dE = dE_i
            max_height = height

    EEc = EE - max_dE/dc*cc
    Ec_ave = np.average(EEc)
    Ec_sig = np.sqrt(np.var(EEc))
    if verbosity > 0: print(f'var: {E_3sig} -> {3*Ec_sig}')
    return Ec_sig/Ec_ave

# set up the energy selection
energy_name = 'energy'
range_name = '40K_1460'

# loop over detectors 
detectors = [f'oppi_{dsp_id}']
store = lh5.Store()
for detector in detectors:
    # get indices for just a selected energy range
    det_db = apdb[detector]
    lh5_group = 'ORSIS3302DecoderForEnergy/raw'
    idx = select_energies(energy_name, range_name, filenames, det_db, lh5_group=lh5_group)

    waveform_name = 'ORSIS3302DecoderForEnergy/raw/waveform/'
    waveforms, _ = store.read_object(waveform_name, filenames, idx=idx)
    print(f'{len(waveforms)} wfs for {detector}')

    # build the table for processing
    tb_data = lh5.Table(col_dict = { 'waveform' : waveforms } )

    # run the optimization
    db_dict = apdb[detector]
    grid_values = run_grid(tb_data, dsp_config, trap_grid, ct_corr_E_var, db_dict=db_dict, verbosity=0)

    # analyze the results
    i_min = np.argmin(grid_values)
    i_min = np.unravel_index(i_min, grid_values.shape)
    print(f'{detector}: min {grid_values[i_min]:.3g} at {ramp_values[i_min[0]]}')

