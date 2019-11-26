#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('clint.mpl')

def main():
    """
    OPPI characterization suite

    pre:
    - put processors in test mode and tweak processor list
    need to tweak:  trap integration + flat top parameters, measure RC constant

    make a function for each one:
    - load t2 file and make energy spectrum
    - run calibration.py on run1
    - measure width of K40 peak (curve_fit: gaussian + linear BG)
    - look at A/E distrbution
    - run calibration.py (separately) and plot the calibration constants
    """
    print("hi")


if __name__=="__main__":
    main()
