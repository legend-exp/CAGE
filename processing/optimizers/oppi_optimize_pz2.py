#!/usr/bin/env python3
import os, sys
import json
import numpy as np
from pprint import pprint
# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt

from pygama.dsp.dsp_optimize import *
from pygama import lh5
from energy_selector import select_energies

if len(sys.argv) < 2:
    print("Usage: oppi_optimize_pz [lh5 raw file(s)]")
    sys.exit()

# get input file, dsp_config, and apdb
filenames = sys.argv[1:]

dsp_id = '06'
with open(os.path.expandvars(f'$CAGE_SW/processing/metadata/dsp/dsp_{dsp_id}.json')) as f:
    dsp_config = json.load(f)
with open('oppi_apdb.json') as f:
    apdb = json.load(f)

# Override dsp_config['outputs'] to contain only what we need for optimization
dsp_config['outputs'] = ['fltp2_sig']
# pprint(dsp_config)

# build a parameter grid for the dsp of choice (double_pole_zero in this case)
pz2_grid = ParGrid()

# -- RC time constant, main component --
# tau1_values = np.linspace(180, 190, 10)
# tau1_values = np.linspace(45, 48, 5)
# tau1_values = np.linspace(50, 53, 8) # campaign 2
tau1_values = np.linspace(50, 53, 4) # campaign 2 -- quick
tau1_values = [ f'{tau:.2f}*us' for tau in tau1_values]
pz2_grid.add_dimension('wf_pz2', 1, tau1_values)

# -- RC time constant #2, fast component --
# tau2_values = np.linspace(2, 6, 5)
# tau2_values = np.linspace(2, 5, 8) # campaign 2
tau2_values = np.linspace(2, 5, 2) # campaign 2 -- quick
tau2_values = [ f'{tau:.2f}*us' for tau in tau2_values]
pz2_grid.add_dimension('wf_pz2', 2, tau2_values)

# -- fraction that time constant #2 should contribute --
# frac_values = np.linspace(0.025, 0.045, 6)
# frac_values = np.linspace(0.2, 0.6, 5) # campaign 2
frac_values = np.linspace(0.2, 0.6, 2) # campaign 2 -- quick
frac_values = [ f'{frac:.3f}' for frac in frac_values]
pz2_grid.add_dimension('wf_pz2', 3, frac_values)

# set up the figure-of-merit to be computed at each grid point
def fltp_sig_mean(tb_out, verbosity):
    mean = np.average(tb_out['fltp2_sig'].nda)
    if verbosity > 1:
        print(f'mean: {mean}')
    return mean

# set up the energy selection
energy_name = 'energy'
range_name = '40K_1460'

ngrid = pz2_grid.get_n_grid_points()
print(f'Setup complete!  ParGrid will simulate {ngrid} points.')

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
    # pprint(db_dict)
    # exit()

    grid_values = run_grid(tb_data, dsp_config, pz2_grid, fltp_sig_mean, db_dict=db_dict, verbosity=1)

    # analyze the results
    i_min = np.argmin(grid_values)
    i_min = np.unravel_index(i_min, grid_values.shape)
    print(f'{detector}: min {grid_values[i_min]:.3f} at {tau1_values[i_min[0]]}, {tau2_values[i_min[1]]}, {frac_values[i_min[2]]}')

    x = np.unravel_index(i_min, grid_values.shape)
    print('grid values:\n', grid_values)
    print('\ni_min:', i_min)
    # print(grid_values[1::])
    # exit()

    # fig, ax = plt.subplots(figsize=(6,5))
    plt.cla()
    plt.plot(tau1_values, fltp_sig_mean, '.r')
    plt.title('Tau 1')
    # plt.savefig('./test.png')
    plt.show()
