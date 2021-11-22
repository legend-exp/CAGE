#!/usr/bin/env python3
import sys, os
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
from functools import partial

from dateutil import parser
from datetime import datetime, timedelta

import pyqtgraph as pg
from pyqtgraph.parametertree import ParameterTree, Parameter
from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import pyqtgraph.console

# have to do this since 'cage' isn't a python package yet
# motor_dir = os.path.expanduser("~/software/CAGE/motors")
# sys.path.append(motor_dir)
# import ..motor_movement as mp


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
    listener.signals.target.connect(cm.dbmon.update_rp)
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
        st2 = MotorMonitor()
        # st2 = QWidget() # blank
        tabs.addTab(st2,"Motor Controller")

        # tab 3 -- detector / DB control.  dragonfly reporting, interlock status, etc.
        # also would like an HV biasing (auto-ramp) & interlock status widget

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

        # set the default endpoints
        # ['cage_pressure', 'cage_coldPlate_temp', 'cage_topHat_temp', 'cage_motor_temp',
        # 'cage_ln_level', 'cage_baseline', 'cage_hv_vmon']
        self.endpts_enabled[10]['value'] = True
        self.endpts_enabled[11]['value'] = True
        self.endpts_enabled[12]['value'] = True
        self.endpts_enabled[13]['value'] = True
        self.endpts_enabled[14]['value'] = True
        self.endpts_enabled[16]['value'] = True
        self.endpts_enabled[17]['value'] = True

        # default time window
        self.t_later = datetime.utcnow()
        self.t_earlier = datetime.utcnow() - timedelta(hours=0.5)

        # create a parameter tree widget from the DB endpoints
        pt_initial = [
            {'name': 'Run Query', 'type': 'group',
             'children': [
               {'name': 'Date (earlier)', 'type':'str', 'value': self.t_earlier.isoformat()},
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
        # reinitialize the plot when the user clicks the "Query DB" button.
        self.rp = RabbitPlot(self.endpts_enabled, self.t_earlier, self.t_later,
                             self.cursor)
        self.p.param('Run Query', 'Query DB').sigActivated.connect(self.update_plot)


        # ---- LAYOUT ----
        # https://doc.qt.io/archives/qt-4.8/qgridlayout.html#public-functions
        # NOTE: addWidget(widget, fromRow, fromColumn, rowSpan, columnSpan)
        self.layout = QGridLayout(self)
        self.setLayout(self.layout)
        self.layout.setColumnStretch(0, 2) # stretch column 0 by 2
        self.layout.setColumnStretch(1, 5)
        self.layout.addWidget(self.pt, 0, 0)
        self.layout.addWidget(self.rp, 0, 1)
        self.show()


    def tree_change(self, param, changes):
        """
        watch for changes in the ParameterTree and update self variables
        """
        for param, change, data in changes:
            path = self.p.childPath(param)
            child_name = '.'.join(path) if path is not None else param.name()

            print(f'  parameter: {child_name}')
            print(f'  change:    {change}')
            print(f'  data:      {data}')

            if child_name == "Run Query.Date (earlier)":
                self.t_earlier = str(data)

            if child_name == "Run Query.Date (later)":
                self.t_later = str(data)

            if "Endpoint Select" in path:
                for i, ept in enumerate(self.endpt_list):
                    if path[-1] == ept:
                        self.endpts_enabled[i]['value'] = data


    def update_plot(self):
        """
        """
        print('got to here')
        self.layout.removeWidget(self.rp)
        self.rp.deleteLater()
        self.rp = RabbitPlot(self.endpts_enabled, self.t_earlier, self.t_later,
                             self.cursor)
        print('made rabbit plot')
        self.layout.addWidget(self.rp, 0,1)
        self.show()


    def update_rp(self, *args):
        """
        called by the main thread's listener function
        """
        # print(args)
        self.rp.update_data(*args)



class MotorMonitor(QWidget):
    """
    MotorMonitor is a set of widgets to control the movement and readout the
    positions of the motors.
    """
    def __init__(self):
        super(QWidget, self).__init__()
        # self.layout = QVBoxLayout(self)
        self.show()

        with open("config.json") as f:
            self.config = json.load(f)
        # ip_address = self.config["encoder_ip"]
        # username = self.config["encoder_usr"]
        # password = self.config["encoder_pwd"]

        # self.pushButton1 = QPushButton("Initialize Motor Control")
        # self.layout.addWidget(self.pushButton1)
        # self.pushButton1.clicked.connect(self.on_motor_clicked)

        pt_motor = [
        {'name': 'Motor Positions', 'type': 'group',
        'children':[
        {'name':'Rotary', 'type':'float', 'value':0},
        {'name':'Linear', 'type':'float', 'value':0},
        {'name':'Source', 'type':'float', 'value':0}
        ]},
        {'name': 'Limit Switch Check', 'type': 'group',
        'children':[
        {'name':'Rotary', 'type':'action'},
        {'name':'Linear', 'type':'action'},
        {'name':'Source', 'type':'action'}
        ]},
        {'name': 'Zero Motors', 'type': 'group',
        'children':[
        {'name': 'WARNING', 'type': 'str', 'value': 'DO NOT CLICK UNLESS \n ASSEMBLY IS LIFTED'},
        {'name':'Rotary', 'type':'action'},
        {'name':'Linear', 'type':'action'},
        {'name':'Source', 'type':'action'}
        ]}
        ]


        self.p = Parameter.create(name='params', type='group', children=pt_motor)

        self.pt = ParameterTree()
        self.pt.setParameters(self.p, showTop=False)

        self.p.sigTreeStateChanged.connect(self.tree_change)

        # def zero_rotary():
        #     zero_rotary_motor()
        # def zero_linear():
        #     zero_linear_motor()
        # def zero_source():
        #     zero_source_motor()
        # def rotary_switch():
        #     rotary_limit_check()
        # def linear_switch():
        #     linear_limit_check()
        # def source_switch():
        #     source_limit_check()

        # self.p.param('Zero Motors', 'Rotary').sigActivated.connect(zero_rotary)
        # self.p.param('Zero Motors', 'Linear').sigActivated.connect(zero_linear)
        # self.p.param('Zero Motors', 'Source').sigActivated.connect(zero_source)
        # self.p.param('Limit Switch Check', 'Rotary').sigActivated.connect(rotary_switch)
        # self.p.param('Limit Switch Check', 'Linear').sigActivated.connect(linear_switch)
        # self.p.param('Limit Switch Check', 'Source').sigActivated.connect(source_switch)

        # self.layout.addWidget(self.pt)

        text = """
                Run mp.movement() to start a movement program.
                Before any move, make sure the motor assembly is lifted off the cold plate.
                Before a new movement routine, make sure the motors are in their zero positions,
                by either clicking the zero motor button, or running through the terminal
                command in mp.movement().
                """
        # namespace = {'pg': pg, 'np': np, 'mp':mp}
        namespace = {'pg': pg, 'np': np}

        self.c = pyqtgraph.console.ConsoleWidget(namespace=namespace, text=text)
        self.c.show()
        self.c.setWindowTitle('Motor Movement Console')

        layout = QGridLayout(self)
        layout.addWidget(self.c,0,1)
        layout.setColumnStretch(1, 2)
        layout.addWidget(self.pt, 0, 0)
        # self.pushButton1.clicked.connect(self.on_motor_clicked)

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


    def on_motor_clicked(self):

        print('Connecting to Motor Controller')
        rotary_limit_check()
        source_limit_check()
        linear_limit_check()


# === RABBITMQ LIVE DB PLOT ====================================================

class RabbitPlot(pg.GraphicsLayoutWidget):
    """
    inherits from GraphicsLayoutWidget to support multiple sub-plots.
    """
    def __init__(self, endpoints, t_earlier=None, t_later=None, db_cursor=None):
        super().__init__()
        self.show()

        self.n_days = 1
        self.n_deque = 10 # should add a check if one exceeds the other
        self.cursor = db_cursor

        # declare endpoints of interest
        self.endpoints = [ept['name'] for ept in endpoints if ept["value"]]
        self.t_earlier = t_earlier
        self.t_later = t_later

        print(self.endpoints)
        exit()

        print("Selected endpoints:", self.endpoints)

        # data for each endpoint goes into circular buffers (aka deques)
        self.deques = {}
        self.plots = {}
        for i, ept in enumerate(self.endpoints):
            self.deques[ept] = collections.deque([], maxlen=self.n_deque)
            self.plots[ept] = self.addPlot(i, 0, title=ept)
            self.plots[ept].showGrid(True, True)
            self.plots[ept].setLogMode(False, False)

        # run the initial DB query
        self.query_db()


    def query_db(self):
        """
        query DB for each endpoint, reset/fill the deques, and plot values.
        """
        pen_colors = ['g', 'r', 'c', 'b', 'm', 'y', 'w']

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
            self.plots[ept].plot(y=yv, x=xv-self.t_offset,
                                 pen=pg.mkPen(pg.intColor(i), width=5))


    def update_data(self, ept=None, xv=None, yv=None):
        """
        every time we get a new value from rabbit, update the plot.
        we have to convert to array because collections.deque doesn't have
        a 'view' function
        """
        if ept in self.endpoints:
            ts = xv.utcnow().timestamp()

            # print(ept, ts, yv)
            # print(ept, len(self.deques[ept]))

            self.deques[ept].append(yv)
            self.deques[ept+"_ts"].append(ts)

            i_ept = self.endpoints.index(ept)

            self.plots[ept].plot(y=np.array(self.deques[ept]),
                                 x=np.array(self.deques[ept+"_ts"])-self.t_offset,
                                 pen=pg.mkPen(pg.intColor(i_ept), width=5))

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
        self.queue_name = f"cage_{np.random.randint(1e6)}" # allow multiple users

        self.channel.exchange_declare(exchange=self.config["exchange"],
                                      exchange_type='topic')

        self.channel.queue_declare(queue=self.queue_name,
                                   exclusive=True)

        # listen to everything that gets posted (.# symbol)
        self.channel.queue_bind(exchange=self.config['exchange'],
                                queue=self.queue_name,
                                routing_key="sensor_value.#")

        self.channel.basic_consume(queue=self.queue_name,
                                   on_message_callback=self.dispatch,
                                   auto_ack=True)

        self.channel.start_consuming()


    def dispatch(self, channel, method, properties, body):
        endpt = method.routing_key.split(".")[-1] # split off "sensor_value."
        record = json.loads(body.decode()) # decode binary string to dict
        xv = parser.parse(record["timestamp"]) # convert to ISO string
        yv = record["payload"]["value_cal"]
        if not isinstance(yv, str):
            self.signals.target.emit(endpt, xv, yv)


class RabbitSignal(QObject):
    """
    used by RabbitListener to communicate w/ the main loop (app.exec_())
    "if you want to define your own signals, they have to be class variables"
    """
    target = pyqtSignal(str, datetime, float)


if __name__=="__main__":
    main()
