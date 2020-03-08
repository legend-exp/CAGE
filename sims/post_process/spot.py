import numpy as np
import h5py
import pandas as pd
import sys
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
plt.style.use('style.mplstyle')

def main():
    
    spot_weighted_hits()

def spot_weighted_hits():

    df =  pd.read_hdf("processed.hdf5", key="procdf")

    m = np.array(df['x'])
    p = np.array(df['y'])

    plt.hist2d(m, p, np.arange(-32,32,0.75), norm=LogNorm())
    plt.xlim(-32,32)
    plt.ylim(-32,32)
    plt.xlabel('x position (mm)', ha='right', x=1.0)
    plt.ylabel('y position (mm)', ha='right', y=1.0)
    plt.title('weighted positional hits at given (x,y)')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Counts')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
        main()
