import numpy as np
from pygama import lh5

def select_energies(energy_name, range_name, filenames, database, lh5_group='', store=None, verbosity=0):
    """
    """
    if energy_name not in database:
        print(f'no energy {energy_name} in database')
        return None
    
    if 'ranges' not in database[energy_name]:
        print(f'database["{energy_name}"] missing field "ranges"')
        return None

    if range_name not in database[energy_name]['ranges']:
        print(f'no range {range_name} in database["{energy_name}"]["ranges"]')
        return None

    E_low = database[energy_name]["ranges"][range_name]["E_low"]
    E_high = database[energy_name]["ranges"][range_name]["E_high"]

    if store is None: store = lh5.Store()

    energies, _ = store.read_object(lh5_group+'/'+energy_name, filenames, verbosity=verbosity)
    return np.where((energies.nda > E_low) & (energies.nda < E_high))

