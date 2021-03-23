import sys, json
import numpy as np
from pygama.dsp.dsp_optimize import *
from pygama import lh5
from energy_selector import select_energies
import os

if len(sys.argv) < 2:
    print("Usage: oppi_optimize_pz [lh5 raw file(s)]")
    sys.exit()

# get input file, dsp_config, and apdb
filenames = sys.argv[1:]
dsp_id = '01'
with open(os.path.expandvars(f'$CAGE_SW/processing/metadata/dsp/dsp_{dsp_id}.json')) as f: dsp_config = json.load(f)
with open('oppi_apdb.json') as f: apdb = json.load(f)

# Override dsp_config['outputs'] to contain only what we need for optimization
dsp_config['outputs'] = ['fltpDCR_sig']

# build a parameter grid for the dsp of choice
pz2_grid = ParGrid()

tau1_values = np.linspace(211, 213, 4)
tau1_values = [ f'{tau:.2f}*us' for tau in tau1_values]
pz2_grid.add_dimension('wf_pzDCR', 1, tau1_values)

tau2_values = np.linspace(4., 6., 4)
tau2_values = [ f'{tau:.2f}*us' for tau in tau2_values]
pz2_grid.add_dimension('wf_pzDCR', 2, tau2_values)

frac_values = np.linspace(0.0435, 0.0455, 3) # 0.035, 0.055, 3
frac_values = [ f'{frac:.3f}' for frac in frac_values]
pz2_grid.add_dimension('wf_pzDCR', 3, frac_values)

# set up the figure-of-merit to be computed at each grid point
def fltp_sig_mean(tb_out, verbosity):
    mean = np.average(tb_out['fltpDCR_sig'].nda)
    if verbosity > 1: print(f'mean: {mean}')
    return mean

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
    grid_values = run_grid(tb_data, dsp_config, pz2_grid, fltp_sig_mean, db_dict=db_dict, verbosity=0)

    # analyze the results
    i_min = np.argmin(grid_values)
    i_min = np.unravel_index(i_min, grid_values.shape)
    print(f'{detector}: min {grid_values[i_min]:.3f} at {tau1_values[i_min[0]]}, {tau2_values[i_min[1]]}, {frac_values[i_min[2]]}')
#     fig, ax = plt.subplots(figsize=(6,5))
#     plt.plot()
    

