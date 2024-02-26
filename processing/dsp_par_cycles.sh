#!/bin/bash

cycles=(4716 4740 4751 4763 4775)
for c in ${cycles[@]}; do
  python gen_dsp_par.py --fdb cage_filedb.lh5 --dl metadata/dataloader_configs/cage_loader_config.json --cyc ${c} --batch;
done
