#!/usr/bin/env python3
import time
import json
import numpy as np
import multiprocessing as mp
from pprint import pprint
import collections
import pika
import psycopg2
from dateutil import parser
from datetime import datetime, timedelta
import pyqtgraph as pg
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLayout, QTabWidget
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    """
    """
    # run normally
    # start_listener()

    # quick test of message passing w/ multiprocessing module
    # test_mp_queue()
    
    # test appending to deque (internal circular buffer of QueueMonitor)
    # test_append()
    
    # run in separate thread with queue for passing messages
    monitor_queue = mp.Queue()
    monitor_thread = mp.Process(target=start_listener, args=(monitor_queue,))
    monitor_thread.start()
    
    while True:
        time.sleep(5)
        val = monitor_queue.get()
        print("got:")
        pprint(val)
    

def test_mp_queue():
    """
    taken from the multiprocessing documentation, about passing data out
    of a queue s/t another process can grab it.
    """
    def f(q):
        q.put([42, None, 'hello'])
        # print("blah") # prints to screen

    q = mp.Queue()
    p = mp.Process(target=f, args=(q,))
    p.start()
    rval = q.get()
    print(type(rval), rval)
    
    # kills process
    p.join() 
    

def start_listener(mpq=None):
    """
    """
    endpoints = ["mj60_baseline"]
    start = "2019-09-30"
    end = "now" 
    qmon = QueueMonitor(endpoints, start, end, mpq=mpq)
    qmon.listen()


class QueueMonitor():
    """
    listen to RabbitMQ and parse each value from the DB: (sensor_value.*)
    the user can specify a list of inputs, or leave blank to save all values
    as they come in.
    
    by default, fill the DB with 'n_days' worth of entries or 'n_list' entries,
    whichever comes first.
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
            self.prefill_deque(self.endpoints, self.val_name)

        
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
        
                
    def prefill_deque(self, endpoints=None, val_name="value_cal"):
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
        one example record: 'sensor_value.mj60_temp'
        {'msgtype': 4,
         'payload': {'value_cal': -192.48926526480463,
                     'value_raw': -192.48926526480463},
         'sender_info': {'commit': 'g7190b92',
                         'exe': '/home/pi/controls/latest/bin/dragonfly',
                         'hostname': 'scannerpi',
                         'package': 'dripline',
                         'service_name': 'scannerpi_service',
                         'username': 'pi',
                         'version': 'v3.7.3'},
         'timestamp': '2019-09-28T23:37:16.592536Z'}
        """
        key = method.routing_key
        record = json.loads(body.decode()) # decode binary string to dict
        # pprint(record)
        
        if self.mpq is not None:
            self.mpq.put(record)
        
        return
        
        # get the name of the record, strip off 'sensor_value.'
        endpoint_name = key.split('.')[-1]
        
        # if endpoint_name not in self.data_lists:
            # self.data_lists[key] = collections.deque(maxlen=5000)
        
        # right now, save only the values and ignore the sender_info
        cal, raw = None, None
        if 'value_cal' in record['payload']:
            cal = record['payload']['value_cal']
        if 'value_raw' in record['payload']:
            raw = record['payload']['value_raw']

        # convert timestamp string to a python friendly type automatically
        ts_raw = record['timestamp']
        ts_fmt = parser.parse(ts_raw)

        # read data to a circular buffer
        for key in self.endpoints:
            
            # initialize the buffer
            if key not in self.data_lists:
                self.data_lists[key] = collections.deque(maxlen=5000)
            
            if cal is not None:
                self.data_lists["cal"].append(cal)


def test_append():
    """
    test record decoding and plotting (no active loop)
    """
    import matplotlib.pyplot as plt
    endpoints = ["mj60_baseline"]
    start = "2019-09-30"
    end = "now" 
    end = '2019-10-03T16:01:56.699883'
    qmon = QueueMonitor(endpoints, start, end)
    
    # a record we would have gotten from 'qmon.listen'
    record = {
        'msgtype': 4,
        'payload': {
            'value_cal': -5.32175, 'value_raw': 1.056375},
        'sender_info': {
            'commit': 'g7190b92',
             'exe': '/home/pi/controls/latest/bin/dragonfly',
             'hostname': 'scannerpi',
             'package': 'dripline',
             'service_name': 'scannerpi_service',
             'username': 'pi',
             'version': 'v3.7.3'},
        'timestamp': '2019-10-03T16:02:20.543470Z'
    }

    # ts = datetime.fromisoformat(record["timestamp"]) # can't handle the "Z"
    ts = parser.parse(record["timestamp"]) # this works
    val = record["payload"]["value_cal"]
    
    # append new values
    qmon.data_lists["mj60_baseline_ts"].append(ts)
    qmon.data_lists["mj60_baseline"].append(val)
    
    # make the plot
    xv = qmon.data_lists["mj60_baseline_ts"]
    yv = qmon.data_lists["mj60_baseline"]
    plt.plot(xv, yv, "-r")

    # superimpose the new point again
    plt.plot(ts, val, ".b", ms='10')

    plt.gcf().autofmt_xdate() # rotates labels
    plt.show()
        

if __name__=="__main__":
    main()