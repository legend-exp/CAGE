#!/usr/bin/env python3
import sys
import time
import json
import argparse
import psycopg2
import collections
import pika
import numpy as np
from pprint import pprint
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from dateutil import parser
from datetime import datetime, timedelta

import pyqtgraph as pg
from pyqtgraph.parametertree import ParameterTree, Parameter
from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


# === PRIMARY EVENT LOOP =======================================================

def main():
    """
    """
    par = argparse.ArgumentParser(description='CAGE Slow Controls App')
    arg, st, sf = par.add_argument, 'store_true', 'store_false'
    arg('-d', '--debug', action=st, help='debug mode')
    args = vars(par.parse_args())
    
    # create the main application
    app = pg.mkQApp()
    pg.setConfigOption('background', 'k')
    pg.setConfigOption('foreground', 'w')
    
    # declare main window
    cm = CAGEMonitor()

    # debug a single plot widget
    # rp = RabbitPlot(["cage_pressure"], "2019-10-04T17:30", "now")
    
    # connect the RabbitPlot emit signal for live updating
    pool = QThreadPool()
    listener = RabbitListener()
    listener.signals.target.connect(cm.dbmon.rp.update_data)
    pool.start(listener)
        
    # start the main Qt loop
    if not args["debug"]:
        exit(app.exec_())
    

# === PRIMARY GUI WINDOW =======================================================

class CAGEMonitor(QMainWindow):
    """
    we use a tabbed main window, to organize widgets for each subsystem.
    
    TODO: add ability to drag tabs to separate windows, and also reattach:
    https://stackoverflow.com/questions/48901854/is-it-possible-to-drag-a-qtabwidget-and-open-a-new-window-containing-whats-in-t
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle('CAGE Detector Monitor')
        self.setGeometry(0, 0, 1200, 800)
        
        tabs = QTabWidget()

        # tab 1 -- db monitor
        self.dbmon = DBMonitor()
        tabs.addTab(self.dbmon,"DB Monitor")
        
        # tab 2 -- motor controller
        # st2 = MotorMonitor()
        # st2 = QWidget() # blank
        # tabs.addTab(st2,"Motor Controller")
        
        # tab 3 -- MJ60 DB monitor?
        # also need an HV biasing & interlock widget

        self.setCentralWidget(tabs)
        self.show()


class DBMonitor(QWidget):
    """
    DBMonitor is a grid of QWidgets, displayed in a tab of CAGEMonitor.
    Available data streams are "endpoints": mj60_baseline, cage_pressure, etc.
    
    TODO: add moveable cross hairs, check the crosshair.py example
    
    TODO: multiple endpoint view.
    since endpoints have different units and y-ranges, let's make it s/t each 
    box that's checked in `Endpoint Select' gets its OWN plot, and the 
    QGridLayout is automatically re-organized to fit more and more plots in 
    column 1, every time the user hits "Query DB".
    """
    def __init__(self):
        super().__init__()
        self.show()
        
        print("Connecting to DB ...")
        
        # establish a postgres connection
        with open("config.json") as f:
            self.config = json.load(f)

        self.connection = psycopg2.connect(
            host = self.config["cage_daq"],
            dbname = self.config["db_name"],
            user = self.config["db_user"],
            password = self.config["password"]
            )
        self.cursor = self.connection.cursor()

        # get a list of all available endpoints in the DB
        self.cursor.execute("SELECT * FROM endpoint_id_map;")
        
        self.endpt_types = {}
        for rec in self.cursor.fetchall():
            self.endpt_types[rec[1]] = rec[2]
        self.endpt_list = [key for key in self.endpt_types]
        
        self.endpts_enabled = []
        for endpt in self.endpt_types:
            self.endpts_enabled.append({'name':endpt, 'type':'bool', 'value':False})
        
        self.endpts_enabled[10]['value'] = True
            
        # default time window of 1 day
        t_later = datetime.utcnow()
        t_earlier = datetime.utcnow() - timedelta(hours=4)
        
        # create a parameter tree widget from the DB endpoints
        pt_initial = [
            {'name': 'Run Query', 'type': 'group', 
             'children': [
               {'name': 'Date (earlier)', 'type':'str', 'value': t_earlier.isoformat()},
               {'name': 'Date (later)', 'type':'str', 'value': "now"},
               {'name': 'Query DB', 'type': 'action'}
            ]},
            {'name': 'Endpoint Select', 'type': 'group', 
             'children': self.endpts_enabled
            }]
        self.p = Parameter.create(name='params', type='group', children=pt_initial)
        self.pt = ParameterTree()
        self.pt.setParameters(self.p, showTop=False)
        
        # connect a simple function
        self.p.sigTreeStateChanged.connect(self.tree_change)

        
        # ---- PLOTTING ----
        
        # -- create a wabbit plot -- 
        # pass our intitial list of endpoints to this
        self.rp = RabbitPlot(self.endpts_enabled, t_earlier, t_later, self.cursor)
        
        # reinitialize the plot when the user clicks the "Query DB" button.
        # TODO: add a flag w/ functools partial to turn live update on/off
        self.p.param('Run Query', 'Query DB').sigActivated.connect(self.rp.__init__)
        
        # could put a second plot with an independent parameter tree here,
        # that listens to the same (or different?) rabbit queue.  

        
        # ---- LAYOUT ----
        # https://doc.qt.io/archives/qt-4.8/qgridlayout.html#public-functions
        # NOTE: addWidget(widget, fromRow, fromColumn, rowSpan, columnSpan)
        layout = QGridLayout(self)
        layout.setColumnStretch(0, 2) # stretch column 0 by 2
        layout.setColumnStretch(1, 5) 
        layout.addWidget(self.pt, 0, 0)
        layout.addWidget(self.rp, 0, 1) 
        # layout.addWidget(plot2, 2, 1, 2, 2)
        self.setLayout(layout)
        
        
    def tree_change(self, param, changes):
        """
        print a message anytime something in the tree changes.
        """
        for param, change, data in changes:
            path = self.p.childPath(param)
            child_name = '.'.join(path) if path is not None else param.name()
            print(f'  parameter: {child_name}')
            print(f'  change:    {change}')
            print(f'  data:      {str(data)}')
            
    

# === RABBITMQ LIVE DB PLOT ====================================================

class RabbitPlot(pg.PlotWidget):
    def __init__(self, endpoints, t_earlier=None, t_later=None, db_cursor=None):
        super().__init__()
        self.show()
        
        self.n_days = 1
        self.n_deque = 10000 # should add a check if one exceeds the other
        self.cursor = db_cursor
        
        # declare endpoints of interest
        self.endpoints = [ept['name'] for ept in endpoints if ept["value"]]
        self.t_earlier = t_earlier
        self.t_later = t_later
        
        # data for each endpoint goes into circular buffers (aka deques)
        self.deques = {}
        self.plots = {}
        
        # set up plot colors (0.0: black, 1.0: white)
        colors = np.arange(0.2, 1.0, len(self.endpoints))
        
        # set up deques and plots
        for i, ept in enumerate(self.endpoints):
            self.deques[ept] = collections.deque([], maxlen=self.n_deque)
            self.plots[ept] = self.plot()
            # self.plots[ept] = self.plot(symbolBrush=(255,255,255), symbolPen='w')
            self.plots[ept].setPen('g', width=3)
        
        self.setLabel('left', 'Value', units="arb. units")
        self.setLabel('bottom', 'Time', units="sec")

        # can we handle this from the UI?
        # self.p1.setLogMode(xMode=False, yMode=True)

        # run the initial DB query
        self.query_db()
        
        
    def query_db(self):
        """
        query DB for each endpoint, reset/fill the deques, and plot values.
        """
        for i, ept in enumerate(self.endpoints):
            str_start = self.t_earlier.isoformat()
            str_end = self.t_later.isoformat()

            # build the query
            query = f"SELECT value_cal, timestamp FROM numeric_data "
            query += f"WHERE endpoint_name='{ept}' "
            query += f"AND timestamp>='{str_start}' and timestamp<='{str_end}';"
            
            print("DB query is:")
            print(query)
            print("")
            self.cursor.execute(query)
            record = self.cursor.fetchall()
            
            # separate value and timestamp. pyqtgraph can't handle datetime objs
            xv = np.array([r[1].timestamp() for r in record])
            yv = np.array([r[0] for r in record])
            self.t_offset = xv[0]
            
            # replace the entire data list with tuples: (value, timestamp)
            self.deques[ept] = collections.deque(yv, maxlen=self.n_deque)
            self.deques[ept + "_ts"] = collections.deque(xv, maxlen=self.n_deque)
            
            # show the plot in pyqtgraph
            self.plots[ept].setData(y=yv, x = xv-self.t_offset)
            
            
    def update_data(self, ept=None, xv=None, yv=None):
        """
        every time we get a new value from rabbit, update the plot
        TODO: need to avoid re-copying the whole array
        https://stackoverflow.com/questions/37079864/reading-live-data-using-pyqtgraph-without-appending-the-data
        """
        if ept in self.endpoints:
            ts = xv.utcnow().timestamp()
            
            self.deques[ept].append(yv)
            self.deques[ept+"_ts"].append(ts)
            
            # this array copy step is bad
            self.plots[ept].setData(y=np.array(self.deques[ept]), 
                                    x=np.array(self.deques[ept+"_ts"]))
            

class RabbitListener(QRunnable):
    """
    uses QRunnable's special 'run' function to start a separate thread with a 
    pika connection that listens for all new messages posted to the DB.
    """
    def __init__(self):
        super().__init__()
        self.signals = RabbitSignal()

    def run(self):
        with open("config.json") as f:
            self.config = json.load(f)
            
        self.cpars = pika.ConnectionParameters(host=self.config['cage_daq'])
        self.conn = pika.BlockingConnection(self.cpars)
        self.channel = self.conn.channel()

        self.channel.exchange_declare(exchange=self.config["exchange"], 
                                      exchange_type='topic')
        
        self.channel.queue_declare(queue=self.config['queue'], 
                                   exclusive=True)
    
        # listen to everything that gets posted (.# symbol)
        self.channel.queue_bind(exchange=self.config['exchange'],
                                queue=self.config['queue'],
                                routing_key="sensor_value.#")

        self.channel.basic_consume(queue=self.config['queue'], 
                                   on_message_callback=self.dispatch, 
                                   auto_ack=True)

        self.channel.start_consuming()


    def dispatch(self, channel, method, properties, body):
        endpt = method.routing_key.split(".")[-1] # split off "sensor_value."
        record = json.loads(body.decode()) # decode binary string to dict
        xv = parser.parse(record["timestamp"]) # convert to ISO string
        yv = record["payload"]["value_cal"]
        self.signals.target.emit(endpt, xv, yv)
    

class RabbitSignal(QObject):
    """
    used by RabbitListener to communicate w/ the main loop (app.exec_())
    "if you want to define your own signals, they have to be class variables"
    """
    target = pyqtSignal(str, datetime, float) 


# === 




if __name__=="__main__":
    main()