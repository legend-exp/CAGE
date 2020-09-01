#!/usr/bin/env python3

import numpy as np
import pyqtgraph as pg

from pyqtgraph.Qt import QtGui, QtCore
from PyQt5.QtWidgets import QMainWindow

def main():
    """
    adapted from the PlotWidget.py example, rearranged to follow Clint's 
    style conventions (main window, etc.)
    https://github.com/pyqtgraph/pyqtgraph/blob/master/examples/PlotWidget.py
    """
    # create the main application
    app = pg.mkQApp()
    pg.setConfigOption('background', 'k')
    pg.setConfigOption('foreground', 'w')
    
    fp = FastPlot()
    
    # start a timer to rapidly update our window.
    # it doesn't look like this can be inside the FastPlot class.
    t = QtCore.QTimer()
    t.timeout.connect(fp.update_data)
    t.start(100) # originally 50.  update interval in ms.
    
    # start the app
    app.exec_()
    
    
class FastPlot(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('updating plot window')
        self.setGeometry(20, 50, 800, 500)
        pw = pg.PlotWidget(name='Plot1') 
        self.setCentralWidget(pw)
        self.show()
        
        self.p1 = pw.plot()
        self.p1.setPen('g')
        pw.setLabel('left', "Value", units="V")
        pw.setLabel('bottom', "Time", units="s")
        # pw.setXRange(0, 2)
        pw.setYRange(0, 1e-10) 
        
    def update_data(self):
        """
        if you need to pass arguments to this, or move it out of the main function, 
        you should do something like this:
        https://stackoverflow.com/questions/13202014/passing-a-parameter-to-qtimer-timeout-signal
        """
        yd, xd = rand(1000)
        self.p1.setData(y=yd, x=xd)
        
    def start_updating(self):
        """
        don't use.
        this doesn't seem to work when it's inside the class
        """
        t = QtCore.QTimer()
        t.timeout.connect(self.update_data)
        t.start(50)
    
    
def rand(n):
    data = np.random.random(n)
    data[int(n*0.1):int(n*0.13)] += .5
    data[int(n*0.18)] += 2
    data[int(n*0.1):int(n*0.13)] *= 5
    data[int(n*0.18)] *= 20
    data *= 1e-12
    return data, np.arange(n, n+len(data)) / float(n)



    
if __name__=="__main__":
    main()
    
