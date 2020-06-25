import sys, os
import numpy as np
from numba import guvectorize
import matplotlib.pyplot as plt

from pygama.dsp.ProcessingChain import ProcessingChain
from pygama.dsp.processors import *
from pygama.dsp.units import *

from pygama.io import io_base as io


def main():

    # h5ls -r filename
    filename = os.path.expandvars("/Volumes/LaCie/Data/CAGE/pygama_dsp/dsp_run11.lh5")

    groupname = "/data"

    lh5 = io.LH5Store()
    data = lh5.read_object(groupname, filename)
    print(data)
    exit()

    energy = data['trapE'].nda
    e1 = data['atrapE'].nda

    # hist, bins = np.histogram(energy, bins=3000,
    #                             range=[0,3000])
    hist1, bins1 = np.histogram(e1, bins=3000,
                                range=[0,3000])
    # plt.semilogy(bins[1:], hist, color='black', ds="steps", linewidth=1.5)
    plt.semilogy(bins1[1:], hist1, color='red', ds="steps", linewidth=1.5)
    plt.show()







if __name__=="__main__":
    main()
