#!/usr/bin/env python3
# typical imports go here:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    """
    docstring
    """
    # func1()
    # func2()
    
    # common stuff
    a, b, c = 1, 2, 3
    
    plot1(a, b, c)
    # plot2(a, b, c)

    
def func1():
    """
    some short note.  this function is analogous to a jupyter notebook cell
    """
    # task 1: get numpy arrays of what you want to plot
    arr1 = [array]
    arr2 = [array]
    
    # now plot
    # plt.plot(arr1, arr2, ".b") # blue dot scatter plot
    
    # plt.hist(arr1) # 1d histogram, matplotlib style
    
    hist, bins = np.histogram(arr1, [options])
    plt.plot(bins[1:], hist, c='r', ls='steps') # red step 1d histogram
    
    # HOMEWORK: np.hist2d.  find an example used in pygama and copy/paste


def func2(a, b, c):
    """
    demoing multi-return
    """
    return a, b, c



if __name__=="__main__":
    main()
