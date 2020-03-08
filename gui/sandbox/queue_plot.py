#!/usr/bin/env python3
import time
import json
import functools
import collections
import datetime
import pika
import psycopg2

import numpy as np
import multiprocessing as mp
import pyqtgraph as pg
from pprint import pprint

from dateutil import parser
from datetime import datetime, timedelta

from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLayout, QTabWidget
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def main():
    """
    """
    # create the main application
    app = pg.mkQApp()
    # pg.setConfigOption('background', 'k')
    # pg.setConfigOption('foreground', 'w')
    
    endpoints = ["cage_pressure"]
    start = "2019-10-04T17:30"
    end = "now" 
    
    # qplot = QueuePlot()

    mpq = mp.Queue()
    mon_thr = mp.Process(target=run_monitor, args=(endpoints, start, end, mpq))
    mon_thr.start()
    
    # while True:
    #     time.sleep(5)
    #     val = mpq.get()
    #     print("got record from queue:")
    #     pprint(val)
    
    # t = QtCore.QTimer()
    # timer_callback = functools.partial(on_timer, mpq)
    # t.timeout.connect(timer_callback)
    # t.start(100)
    
    # start the app
    app.exec_()
    exit()
    

def run_monitor(endpoints, start=None, end=None, mpq=None):
    qmon = RabbitMonitor(endpoints, start, end, mpq=mpq)
    qmon.listen()


def on_timer(mpq):
    """
    """
    # update the deque's in qmon.data_lists
    print("i'm here")
    record = mpq.get()
    print("got one!")
    pprint(record)

    # val_last = qmon.data_lists["cage_pressure"][-1]
    # val = val_last * np.random.uniform(0.9, 1.1)
    
    # qmon.data_lists["cage_pressure"].append(val)
    
    # dt = datetime.utcnow()
    # qmon.data_lists["cage_pressure_ts"].append(dt)

    # qplot.update_data(qmon)


class QueuePlot(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('updating plot window')
        self.setGeometry(20, 50, 800, 500)
        pw = pg.PlotWidget(name='Plot1') 
        self.setCentralWidget(pw)
        self.show()
        
        self.p1 = pw.plot()
        self.p1.setPen('g')
        # self.p1.setLogMode(xMode=False, yMode=True)
        pw.setLabel('left', "Value", units="V")
        pw.setLabel('bottom', "Time", units="s")
        # pw.setXRange(0, 2)
        # pw.setYRange(0, 100) 
        

    def update_data(self, qmon):
        """
        """
        yv = np.array(qmon.data_lists["cage_pressure"])
        
        # pyqtgraph doesn't handle datetime objects, so use unix times for now
        xv = qmon.data_lists["cage_pressure_ts"]
        x0 = int(xv[0].strftime("%s"))
        xv = np.array([int(x.strftime("%s")) - x0 for x in xv])

        self.p1.setData(y=yv, x=xv)
        

class RabbitMonitor():
    """
    listen to RabbitMQ and parse each value from the DB: (sensor_value.*)
    the user can specify a list of inputs, or leave blank to save all values
    as they come in.
    """
    def __init__(self, endpoints=None, start_date=None,
                 end_date=None, val_name="value_cal", mpq=None):
        with open('config.json') as f:
            self.config = json.load(f)

        # defaults
        self.n_days = 1 
        self.n_list = 10000

        # use pika to connect to the message queue
        self.cpars = pika.ConnectionParameters(host=self.config['cage_daq'])
        self.conn = pika.BlockingConnection(self.cpars)
        self.channel = self.conn.channel()
        self.channel.exchange_declare(exchange='cage', exchange_type='topic')
        
        # multiprocessing queue for passing messages out of the thread
        self.mpq = mpq
        
        # circular buffers to store data from each sensor (key) of interest
        self.endpoints = endpoints
        self.val_name = val_name # for now, assume same for each endpoint
        self.data_lists = {}

        # optionally pre-fill the buffers
        self.conn = None
        self.end = self.get_datetime(end_date)
        if start_date is None:
            self.start = self.end - timedelta(days=self.n_days)
        else:
            self.start = self.get_datetime(start_date)
        if (start_date is not None) or (end_date is not None):
            self.prefill_deques(self.endpoints, self.val_name)

        
    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        self.conn.close()

        
    def get_datetime(self, date_str, end=False):
        """
        take an input string from the user and parse to a datetime obj
        """
        if date_str == "now" or date_str is None:
            return datetime.utcnow()
        try:
            return parser.parse(date_str)
        except:
            print("failed to parse date string:", date_str)
            exit()
        
                
    def prefill_deques(self, endpoints=None, val_name="value_cal"):
        """
        pre-fill the data lists with a postgres call
        """
        self.conn = psycopg2.connect(
            dbname = self.config["db_name"],
            user = self.config["db_user"], 
            password = self.config["password"],
            host = self.config["cage_daq"]
            )
        cursor = self.conn.cursor()        
        
        # save all endpoint names by default
        if endpoints is None:
            cmd = "SELECT * FROM endpoint_id_map;"
            cursor.execute(cmd)
            record = cursor.fetchall()
            self.endpoints = [rec[1] for rec in record]
        else:
            self.endpoints = [f'{key}' for key in endpoints]
            
        for key in self.endpoints:
            # need to get an ISO-formatted string from self.start and self.end
            str_start = self.start.isoformat()
            str_end = self.end.isoformat()

            # build the query
            query = f"SELECT {val_name}, timestamp FROM numeric_data "
            query += f"WHERE endpoint_name='{key}' "
            query += f"AND timestamp>='{str_start}' and timestamp<='{str_end}';"
            
            print("query is:")
            print(query)
            print("")
            
            cursor.execute(query)
            record = cursor.fetchall()
            
            # separate value and timestamp (not using dataframe b/c appending)
            xv = [r[1] for r in record]
            yv = [r[0] for r in record]
            
            # replace the entire data list with tuples: (value, timestamp)
            self.data_lists[key] = collections.deque(yv, maxlen=self.n_list)
            self.data_lists[key + "_ts"] = collections.deque(xv, maxlen=self.n_list)
            
    
    def listen(self):
        """
        using the pika BlockingConnection, we listen to the queue.  when we
        get a value, we run the callback function decode_values.
        """
        result = self.channel.queue_declare(queue=self.config['queue'], 
                                            exclusive=True)
        if self.endpoints is not None:
            for key in self.endpoints:
                self.channel.queue_bind(exchange=self.config['exchange'], 
                                        queue=self.config['queue'],
                                        routing_key=f"sensor_value.{key}")
        else:
            self.channel.queue_bind(exchange=self.config['exchange'],
                                    queue=self.config['queue'],
                                    routing_key="sensor_value.#")
                                
        self.channel.basic_consume(queue=self.config['queue'], 
                                   on_message_callback=self.decode_values, 
                                   auto_ack=True)

        # starts a while-type loop
        print("wabbit eatin hay")
        self.channel.start_consuming()
        
        
    def decode_values(self, ch, method, properties, body):
        """
        decode the DB records and save to results arrays.  
        """
        key = method.routing_key
        record = json.loads(body.decode()) # decode binary string to dict
        
        # pprint(record)
        print('got a record')
        
        if self.mpq is not None:
            self.mpq.put(record)
        

if __name__=="__main__":
    main()
    
