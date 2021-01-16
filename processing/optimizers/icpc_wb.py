import json
import matplotlib.pyplot as plt
from pygama.dsp.WaveformBrowser import WaveformBrowser
from energy_selector import select_energies

filenames = [#'raw/Run70053_raw.lh5',
             'raw/Run70054_raw.lh5', ]
detector = 'icpc1'

with open('icpc_apdb.json') as f: apdb = json.load(f)

# set up the energy selection
lh5_group = 'icpcs/'+detector+'/raw/'
idx = select_energies('energy', '208Tl_2615', filenames, apdb[detector], lh5_group=lh5_group)
print(len(idx[0]), 'waveforms')

wb = WaveformBrowser(filenames,
                     lh5_group,
                     dsp_config='icpc_dsp.json',
                     database=apdb[detector],
                     #waveforms=['waveform', 'wf_pz', 'wf_pz2', 'wf_trap', 'curr10', 'wf_atrap'],
                     waveforms=['waveform', 'wf_pz', 'wf_pz2'],
                     #waveforms=['wf_pz', 'wf_pz2', 'wf_trap', 'wf_atrap'],
                     #lines=['bl', 'tp_0', 'tp_max'], legend=['tp_0'])
                     lines=['fltp2'],#, 'fltp2'],
                     #lines=['trapEftp', 'tp_0', 'tp_max'],
                     #legend=['tp_0'],
                     selection=list(idx[0]), # wf browser doesn't take numpy-like index tuples
                     verbosity=1)

wb.draw_next(2)
plt.show()
