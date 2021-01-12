#!/usr/bin/env python3
import pandas as pd
import pygama.io.lh5 as lh5

fin = '/global/project/projectdirs/legend/users/gothman/CAGE'
fin += '/dsp/cage_run110_cyc1186_dsp.lh5' # run 110 is cycles 1184--1190

name = 'ORSIS3302DecoderForEnergy/dsp'

energy = lh5.load_nda([fin], ['trapEmax'], name)['trapEmax']
energy = pd.Series(energy)

print(energy, len(energy))
# print(energy.loc[(energy > 3597) & (energy < 3617)])
emask = (energy > 3597) & (energy < 3617)
print(emask.value_counts())
# print(energy.max())


# sto = lh5.Store()
# tb, n = sto.read_object(name, fin)