#!/usr/bin/env python3
import argparse
import numpy as np

import pyqtgraph as pg
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import pyqtSlot

import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType


def main():
    """
    """
    # parse user args
    par = argparse.ArgumentParser(description='Live plotting app')
    arg, st, sf = par.add_argument, 'store_true', 'store_false'
    arg('-d', '--debug', action=st, help='debug mode')
    args = vars(par.parse_args())

    # Create the main application instance
    app = pg.mkQApp()
    pg.setConfigOption('background', 'k')
    pg.setConfigOption('foreground', 'w')
    
    # run examples
    # simple_plot()
    # pw = GridPlot()
    # sg = SimpleGrid()
    
    # pt = ParamTree()
    # pt.do_stuff()
    
    st = SimpleTree()
    st.do_stuff()
    
    # start Qt event loop and wait to quit
    if not args["debug"]:
        exit(app.exec_())
    

def simple_plot():
    """
    show two curves in a PlotWindow w/o any extra Qt machinery or classes needed
    this is the only one that seems to work when defined only in a function
    """
    # make some fake data
    n = 1000
    xv = np.arange(n)
    yv = 1 * pg.gaussianFilter(np.random.random(size=n), 10)
    yv += 1 * np.random.random(size=n)
    yv2 = yv * np.random.random(size=n)
    yv3 = yv * np.random.random(size=n)
    
    # plot the curves
    plotWindow = pg.plot(title="Three plot curves")
    y = [yv, yv2, yv3]
    for i in range(3):
        # setting pen=(i,3) automatically creates three different-colored pens
        plotWindow.plot(xv, y[i], pen=(i,3)) 


class GridPlot(pg.GraphicsWindow):
    """
    -- if you want to specify the window away from the main loop, 
    you have to put it into a class object that can stay in memory.
    a simple function call just runs and doesn't display anything.
    
    -- after calling self.show once, calls to plot will update the 
    plot automatically.
    """
    def __init__(self):
        super().__init__()
        p0 = self.addPlot(row=0, col=0)
        p1 = self.addPlot(row=0, col=1)
        p2 = self.addPlot(row=1, col=0, colspan=2)
        self.show()
        
        n = 1000
        xv = np.arange(n)
        yv = 1 * pg.gaussianFilter(np.random.random(size=n), 10)
        yv += 1 * np.random.random(size=n)
        yv2 = yv * np.random.random(size=n)
        yv3 = yv * np.random.random(size=n)
        
        p0.plot(xv, yv, pen='r')
        p1.plot(xv, yv2, pen='g')
        p2.plot(xv, yv2, pen='b')
        
        # a second call to plot will update it, again without calling show
        p2.plot(xv, yv2 - 20, pen='w')


class SimpleGrid(QWidget):
    """
    this looks closer to an interface w/ a live plot
    """
    def __init__(self):
        super().__init__()
        
        # Define a top-level widget to hold everything
        # w = QtGui.QWidget()
        
        # Create some widgets to be placed inside
        btn = QtGui.QPushButton('press me')
        text = QtGui.QLineEdit('enter text')
        listw = QtGui.QListWidget()
        plot = pg.PlotWidget()
        
        # connect a keyboard shortcut to a member function that can do stuff
        label = QtGui.QLabel("hit q to quit", self)
        shortcut = QtGui.QShortcut(QtGui.QKeySequence("q"), self)
        shortcut.activated.connect(self.quitter)
        
        # plot some data
        n = 1000
        xv = np.arange(n)
        yv = 1 * pg.gaussianFilter(np.random.random(size=n), 10)
        plot.plot(xv, yv, pen='r')
        
        # Create a grid layout to manage the widgets size and position
        layout = QtGui.QGridLayout()
        self.setLayout(layout)
        
        # Add widgets to the layout in their proper positions
        layout.addWidget(btn, 0, 0)   # button goes in upper-left
        layout.addWidget(text, 1, 0)   # text edit goes in middle-left
        layout.addWidget(listw, 2, 0)  # list widget goes in bottom-left
        layout.addWidget(label, 3, 0) # note about keyboard shortcut on bottom
        layout.addWidget(plot, 0, 1, 3, 1)  # plot goes on right side, spanning 3 rows
        
        # Display the widget as a new window
        self.show()
        
    @pyqtSlot()
    def quitter(self):
        print("Quitting!")
        exit()


class ParamTree(QWidget):
    """
    adapted from:
    https://github.com/pyqtgraph/pyqtgraph/blob/develop/examples/parametertree.py
    """
    def __init__(self):
        super().__init__()
        self.show()
        self.resize(800,1000)
        # self.resize(self.sizeHint()) # this might work better when embedded
        
    def do_stuff(self):
        
        # NOTE: I could probably make this into a json configuration file
        params = [
        
            # item 1
            {'name': 'Basic parameter data types', 'type': 'group', 'children': [
                {'name': 'Integer', 'type': 'int', 'value': 10},
                {'name': 'Float', 'type': 'float', 'value': 10.5, 'step': 0.1},
                {'name': 'String', 'type': 'str', 'value': "hi"},
                {'name': 'List', 'type': 'list', 'values': [1,2,3], 'value': 2},
                {'name': 'Named List', 'type': 'list', 'values': {"one": 1, "two": "twosies", "three": [3,3,3]}, 'value': 2},
                {'name': 'Boolean', 'type': 'bool', 'value': True, 'tip': "This is a checkbox"},
                {'name': 'Color', 'type': 'color', 'value': "FF0", 'tip': "This is a color button"},
                {'name': 'Gradient', 'type': 'colormap'},
                {'name': 'Subgroup', 'type': 'group', 'children': [
                {'name': 'Sub-param 1', 'type': 'int', 'value': 10},
                {'name': 'Sub-param 2', 'type': 'float', 'value': 1.2e6},
                ]},
                {'name': 'Text Parameter', 'type': 'text', 'value': 'Some text...'},
                {'name': 'Action Parameter', 'type': 'action'},
                ]},
            
            # item 2
            {'name': 'Numerical Parameter Options', 'type': 'group', 'children': [
                {'name': 'Units + SI prefix', 'type': 'float', 'value': 1.2e-6, 'step': 1e-6, 'siPrefix': True, 'suffix': 'V'},
                {'name': 'Limits (min=7;max=15)', 'type': 'int', 'value': 11, 'limits': (7, 15), 'default': -6},
                {'name': 'DEC stepping', 'type': 'float', 'value': 1.2e6, 'dec': True, 'step': 1, 'siPrefix': True, 'suffix': 'Hz'},
            ]},
            
            # item 3
            {'name': 'Save/Restore functionality', 'type': 'group', 'children': [
            {'name': 'Save State', 'type': 'action'},
            {'name': 'Restore State', 'type': 'action', 'children': [
            {'name': 'Add missing items', 'type': 'bool', 'value': True},
            {'name': 'Remove extra items', 'type': 'bool', 'value': True},
            ]},
            ]},
            
            # item 4
            {'name': 'Extra Parameter Options', 'type': 'group', 'children': [
            {'name': 'Read-only', 'type': 'float', 'value': 1.2e6, 'siPrefix': True, 'suffix': 'Hz', 'readonly': True},
            {'name': 'Renamable', 'type': 'float', 'value': 1.2e6, 'siPrefix': True, 'suffix': 'Hz', 'renamable': True},
            {'name': 'Removable', 'type': 'float', 'value': 1.2e6, 'siPrefix': True, 'suffix': 'Hz', 'removable': True},
            ]},
            
            # item 5
            ComplexParameter(name='Custom parameter group (reciprocal values)'),
            ScalableGroup(name="Expandable Parameter Group", children=[
            {'name': 'ScalableParam 1', 'type': 'str', 'value': "default param 1"},
            {'name': 'ScalableParam 2', 'type': 'str', 'value': "default param 2"},
            ]),
            ]
        
        # Create tree of Parameter objects
        p = Parameter.create(name='params', type='group', children=params)
        
        # If anything changes in the tree, print a message
        def change(param, changes):
            print("tree changes:")
            for param, change, data in changes:
                path = p.childPath(param)
                if path is not None:
                    childName = '.'.join(path)
                else:
                    childName = param.name()
                print('  parameter: %s'% childName)
                print('  change:    %s'% change)
                print('  data:      %s'% str(data))
                print('  ----------')
            
        p.sigTreeStateChanged.connect(change)

        # # print messages to terminal when anything changes
        # def valueChanging(param, value):
        #     print("Value changing (not finalized): %s %s" % (param, value))
        # for child in p.children():
        #     child.sigValueChanging.connect(valueChanging)
        #     for ch2 in child.children():
        #         ch2.sigValueChanging.connect(valueChanging)

        def save():
            global state
            state = p.saveState()
            
        def restore():
            global state
            add = p['Save/Restore functionality', 'Restore State', 'Add missing items']
            rem = p['Save/Restore functionality', 'Restore State', 'Remove extra items']
            p.restoreState(state, addChildren=add, removeChildren=rem)
        p.param('Save/Restore functionality', 'Save State').sigActivated.connect(save)
        p.param('Save/Restore functionality', 'Restore State').sigActivated.connect(restore)


        # Create two ParameterTree widgets, both accessing the same data
        t = ParameterTree()
        t.setParameters(p, showTop=False)
        t.setWindowTitle('pyqtgraph example: Parameter Tree')
        t2 = ParameterTree()
        t2.setParameters(p, showTop=False)

        # win = QtGui.QWidget()
        layout = QtGui.QGridLayout()
        
        self.setLayout(layout)
        layout.addWidget(QtGui.QLabel("These are two views of the same data. They should always display the same values."), 0,  0, 1, 2)
        layout.addWidget(t, 1, 0, 1, 1)
        layout.addWidget(t2, 1, 1, 1, 1)

        ## test save/restore
        s = p.saveState()
        p.restoreState(s)


class ComplexParameter(pTypes.GroupParameter):
    """
    adapted from:
    https://github.com/pyqtgraph/pyqtgraph/blob/develop/examples/parametertree.py
    Test subclassing parameters
    This parameter automatically generates two child parameters which are always reciprocals of each other
    """
    def __init__(self, **opts):
        opts['type'] = 'bool'
        opts['value'] = True
        pTypes.GroupParameter.__init__(self, **opts)
        
        self.addChild(
            {'name': 'A = 1/B', 'type': 'float', 'value': 7, 'suffix': 'Hz', 
             'siPrefix': True})
        self.addChild(
            {'name': 'B = 1/A', 'type': 'float', 'value': 1/7., 'suffix': 's', 
             'siPrefix': True})
        self.a = self.param('A = 1/B')
        self.b = self.param('B = 1/A')
        self.a.sigValueChanged.connect(self.aChanged)
        self.b.sigValueChanged.connect(self.bChanged)
        
    def aChanged(self):
        self.b.setValue(1.0 / self.a.value(), blockSignal=self.bChanged)

    def bChanged(self):
        self.a.setValue(1.0 / self.b.value(), blockSignal=self.aChanged)


class ScalableGroup(pTypes.GroupParameter):
    """
    adapted from:
    https://github.com/pyqtgraph/pyqtgraph/blob/develop/examples/parametertree.py
    
    test add/remove
    this group includes a menu allowing the user to add new parameters into its child list
    """
    def __init__(self, **opts):
        opts['type'] = 'group'
        opts['addText'] = "Add"
        opts['addList'] = ['str', 'float', 'int']
        pTypes.GroupParameter.__init__(self, **opts)
    
    def addNew(self, typ):
        val = {'str': '', 'float': 0.0, 'int': 0}[typ]
        self.addChild(dict(name="ScalableParam %d" % (len(self.childs)+1), 
                           type=typ, value=val, removable=True, renamable=True))

    
class SimpleTree(QWidget):
    def __init__(self):
        super().__init__()
        self.show()
        self.resize(1000,1000)
        self.tree = ParameterTree()
        self.tree.setWindowTitle('data monitor thing')
        
        self.params = [
            {'name': 'Basic parameter data types', 
             'type': 'group', 
             'children': [
                 {'name': 'Integer', 'type': 'int', 'value': 10},
                 {'name': 'Float', 'type': 'float', 'value': 10.5, 'step': 0.1},
                 {'name': 'String', 'type': 'str', 'value': "hi"},
                 {'name': 'List', 'type': 'list', 'values': [1,2,3], 'value': 2},
                 {'name': 'Named List', 'type': 'list', 
                  'values': {"one": 1, "two": "twosies", "three": [3,3,3]}, 
                  'value': 2},
                 {'name': 'Boolean', 'type': 'bool', 'value': True, 
                  'tip': "This is a checkbox"},
                 # {'name': 'Gradient', 'type': 'colormap'},
                 {'name': 'Subgroup', 
                  'type': 'group', 
                  'children': [
                      {'name': 'Sub-param 1', 'type': 'int', 'value': 10},
                      {'name': 'Sub-param 2', 'type': 'float', 'value': 1.2e6}
                      ]
                  },
                 {'name': 'Text Parameter', 'type': 'text', 'value': 'text here'},
                 {'name': 'Action Parameter', 'type': 'action'}
              ]
             }
            ]
        
        
    def do_stuff(self):
        
        p = Parameter.create(name='params', type='group', children=self.params)
        
        def change(param, changes):
            print("tree changes:")
            for param, change, data in changes:
                path = p.childPath(param)
                if path is not None:
                    childName = '.'.join(path)
                else:
                    childName = param.name()
                print('  parameter: %s'% childName)
                print('  change:    %s'% change)
                print('  data:      %s'% str(data))
                print('  ----------')
        p.sigTreeStateChanged.connect(change)

        # create a parameter tree widget
        t = ParameterTree()
        t.setParameters(p, showTop=False)
        t.setWindowTitle('pyqtgraph example: Parameter Tree')
        
        # plot some data
        plot = pg.PlotWidget()
        n = 1000
        xv = np.arange(n)
        yv = 1 * pg.gaussianFilter(np.random.random(size=n), 10)
        plot.plot(xv, yv, pen='r')
        
        # make a second plot
        plot2 = pg.PlotWidget()
        plot2.plot(xv, yv, pen='g')
        
        # set the layout of the widget
        layout = QtGui.QGridLayout()
        
        # layout.columnStretch(5)
        layout.setColumnStretch(2, 2)
        
        # NOTE: (widget, # y_row, x_row, y_span, x_span)
        # layout.addWidget(QtGui.QLabel("Data monitor thing"), 0, 0, 1, 2)
        
        layout.addWidget(t, 0, 0, 2, 1)
        
        layout.addWidget(plot, 0, 2) 
        layout.addWidget(plot2, 1, 2)
        
        self.setLayout(layout)

        ## test save/restore
        # s = p.saveState()
        # p.restoreState(s)
        
        

    
if __name__=="__main__":
    main()