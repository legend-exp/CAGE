import sys, h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pygama.io.io_base as io


def main():
    filename = '/Users/gothman/Data/CAGE/pygama_dsp/dsp_run42.lh5'
    plot_spectrum(filename)


def plot_spectrum(filename):
    lh5 = io.LH5Store()
    df = lh5.read_object('data', filename).get_dataframe()
    df['trapE'].plot.hist(bins=1000)
    plt.show()










if __name__ == '__main__':
	main()
