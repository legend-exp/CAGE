import numpy as np
import h5py
import pandas as pd
import sys
from decimal import *
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
plt.style.use('style.mplstyle')

# a value is where the exponential part of dead layer function becomes linear.
# b value is where the dead layer ends.
# sigma value defines how fast exponential part falls.

if(len(sys.argv) != 5):
    print('Usage: DL_function.py [a value] [b value] [sigma] [surface (passivated or outer)]')
    sys.exit()

a = float(sys.argv[1])
b = float(sys.argv[2])
sigma = float(sys.argv[3])

d = r'$f_{1}(x)=Ae^{\sigma x}+B$'
f = r'$f_{2}(x)=Mx+C$'
g = r'$f_{1}(a)=f_{2}(a)$'
h = r'$f_{1}(0)=0$'
j = r'$f_{2}(b)=1$'
k = 'a='+sys.argv[1]+', b='+sys.argv[2]+', sigma='+sys.argv[3]+'\n'

colors = ['black', 'blue','white', 'white', 'white', 'white']
lines = [Line2D([0], [0], color=c, lw=4) for c in colors] 
labels = [d, f, g, h, j, k]

r1_vals = np.linspace(0,a,1000)
f1 = 1-(np.exp(sigma*(a-r1_vals))/(-1+np.exp(sigma*a)-a*sigma+b*sigma)+(1+a*sigma-b*sigma)/(1-np.exp(sigma*a)+a*sigma-b*sigma))
r2_vals = np.linspace(a,b,1000)
f2 = 1-(sigma*r2_vals/(1-np.exp(a*sigma)+a*sigma-b*sigma)+b*sigma/(-1+np.exp(a*sigma)-a*sigma+b*sigma))
plt.plot(r1_vals, f1, '-k')
plt.plot(r2_vals,f2, '-b')
plt.xlim(0,b)
plt.ylim(0,plt.ylim()[1])
plt.title('Detector Activeness ('+sys.argv[4]+' Surface)')
plt.xlabel("Depth (mm)", ha='right', x=1.0)
plt.ylabel("Activeness", ha='right', y=1.0)
plt.legend(lines, labels, frameon=False, loc='upper left', fontsize='small')
plt.tight_layout()
plt.show()
