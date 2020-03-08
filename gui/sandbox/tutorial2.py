#!/usr/bin/env python3
import sys
import argparse
import json
import multiprocessing
import pika
import psycopg2
import numpy as np
from pprint import pprint
import pyqtgraph as pg
from pyqtgraph.parametertree import ParameterTree, Parameter
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from PyQt5 import QtGui
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLayout, QTabWidget

def main():
    """
    """
    par = argparse.ArgumentParser(description='Live plotting app')
    arg, st, sf = par.add_argument, 'store_true', 'store_false'
    arg('-d', '--debug', action=st, help='debug mode')
    args = vars(par.parse_args())
    
    read_queue()
    exit()
    # read_db()

    # Create the main application
    app = pg.mkQApp()
    pg.setConfigOption('background', 'k')
    pg.setConfigOption('foreground', 'w')
    
    # Show the main window
    dm = DataMonitor()
    
    if not args["debug"]:
        exit(app.exec_())
    

class DataMonitor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('CAGE Detector Monitor')
        self.setGeometry(0, 0, 1000, 900)
        
        # put a single widget in the main window
        # st = SimpleTree()
        # st.do_stuff()
        # self.setCentralWidget(st)
        
        # put a tabbed widget in the main window
        tabs = QTabWidget()

        # tab 1 -- db monitor
        st = SimpleTree()
        st.do_stuff()
        tabs.addTab(st,"DB Monitor")
        
        # tab 2 -- motor controller
        st2 = SimpleTree()
        tabs.addTab(st2,"Motor Controller")

        self.setCentralWidget(tabs)
        
        self.show()
        

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

        # test save/restore
        s = p.saveState()
        p.restoreState(s)
        

def read_queue():
    """
    example of accessing the live messages from RabbitMQ
    """
    with open("config.json") as f:
        config = json.load(f)
    
    cpars = pika.ConnectionParameters(host=config["cage_daq"])
    connection = pika.BlockingConnection(cpars)
    channel = connection.channel()

    channel.exchange_declare(exchange=config["exchange"], exchange_type='topic')
    
    channel.queue_declare(queue=config["queue"], exclusive=True)
    
    channel.queue_bind(queue=config["queue"], exchange=config["exchange"], 
                       routing_key='sensor_value.#')
    
    channel.basic_consume(queue=config["queue"], on_message_callback=callback, 
                          auto_ack=True)

    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()


def callback(ch, method, properties, body):
    """
    the callback function can be defined inside or outside read_queue,
    and can also be a member of 'self' when defined as part of a class
    """
    record = json.loads(body.decode()) # decode binary string to dict
    pprint(record)

    
def read_db():
    """
    example of accessing the SQL db history
    
    current endpoints:
    (1, 'mj60_temp', 'numeric')
    (2, 'mj60_baseline', 'numeric')
    (3, 'mj60_pressure', 'numeric')
    (4, 'mj60_hv_vset', 'numeric')
    (5, 'mj60_hv_vmon', 'numeric')
    (6, 'mj60_hv_imon', 'numeric')
    (7, 'mj60_hv_rup', 'numeric')
    (8, 'mj60_hv_rdown', 'numeric')
    (9, 'mj60_hv_status', 'string')
    (10, 'mj60_ln_level', 'numeric')
    (11, 'cage_pressure', 'numeric')
    (12, 'cage_coldPlate_temp', 'numeric')
    (13, 'cage_ln_level', 'numeric')
    (14, 'cage_motor_temp', 'numeric')
    (15, 'cage_topHat_temp', 'numeric')
    """
    with open("config.json") as f:
        config = json.load(f)
    
    conn = psycopg2.connect(dbname='cage_sc_db', user='cage_db_user', 
                            password='legend', host='10.66.193.71')
    cursor = conn.cursor()

    # cmd = "SELECT value_raw, timestamp FROM numeric_data WHERE endpoint_name='mj60_baseline' AND timestamp>'2019-09-27T00:00';"
    
    # cmd = "SELECT * FROM endpoint_id_map;"
    
    # cmd = "SELECT value_cal, timestamp FROM numeric_data WHERE endpoint_name='cage_coldPlate_temp' AND timestamp>'2019-09-03T00:02';"
    
    # cmd = "SELECT value_cal, timestamp FROM numeric_data WHERE endpoint_name='cage_pressure' AND timestamp>'2019-09-27T00:00';"
    
    cmd = "SELECT value_cal, timestamp FROM numeric_data WHERE endpoint_name='cage_ln_level' AND timestamp>'2019-09-27T00:00';"
    
    # cmd = "SELECT value_raw, timestamp FROM string_data WHERE endpoint_name='mj60_hv_status' AND timestamp>'2019-08-01';"
    
    cursor.execute(cmd)

    # retrieve data.  returns a list of tuples.
    record = cursor.fetchall()
    
    # print(type(record[0]))
    
    # dt = record[0][1]
    
    # print(dt)
    
    for rec in record:
        print(rec)
        

if __name__=="__main__":
    main()
