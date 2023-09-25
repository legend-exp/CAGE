#!/bin/bash

cycles=(4430 4442 4455);
for c in ${cycles[@]}; do
  python gen_dsp_par.py --fdb cage_filedb.lh5 --dl metadata/dataloader_configs/cage_loader_config.json --cyc ${c} --batch;
done