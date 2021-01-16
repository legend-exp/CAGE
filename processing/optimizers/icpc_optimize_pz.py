import sys, json
import numpy as np
from pygama.dsp.dsp_optimize import *
from pygama import lh5

if len(sys.argv) < 2:
    print("Usage: icpc_optimize_pz [lh5 raw file]")
    sys.exit()

# get input file and dsp_config
filename = sys.argv[1]
with open('icpc_dsp.json') as f: dsp_config = json.load(f)

# Override dsp_config['outputs'] to contain only what we need for optimization
dsp_config['outputs'] = ['fltp_sig']

# build a parameter grid for the dsp of choice
tau_grid = ParGrid()
tau_values = np.linspace(72, 75, 31)
tau_values = [ f'{tau}*us' for tau in tau_values]
tau_grid.add_dimension('wf_pz', 1, tau_values)

# set up the figure-of-merit to be computed at each grid point
def fltp_sig_mean(tb_out, verbosity):
    mean = np.average(tb_out['fltp_sig'].nda)
    if verbosity > 1: print(f'mean: {mean}')
    return mean


# loop over detectors 
detectors = [ 'icpc1', 'icpc2', 'icpc3', 'icpc4' ]
store = lh5.Store()
n_rows = 500
for detector in detectors:

    # pull out waveforms only for a selected energy range
    energy_name = 'icpcs/' + detector + '/raw/energy'
    energies, _ = store.read_object(energy_name, filename)
    idx = np.where(energies.nda > 2000)
    waveform_name = 'icpcs/' + detector + '/raw/waveform'
    waveforms, _ = store.read_object(waveform_name, filename, n_rows=n_rows, idx=idx)
    print(f'{len(waveforms)} wfs for {detector}')

    # build the table for processing
    tb_data = lh5.Table(col_dict = { 'waveform' : waveforms } )

    # run the optimization
    grid_values = run_grid(tb_data, dsp_config, tau_grid, fltp_sig_mean, verbosity=0)

    # analyze the results
    i_min = np.argmin(grid_values)
    print(f'{detector}: min at {tau_values[i_min]}')

