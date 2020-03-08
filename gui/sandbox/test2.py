import sys
import json
import pika
from pprint import pprint
import pyqtgraph as pg
import numpy as np

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

def main():
    """
    finally, a working example of starting a RabbitMQ listener and using it to
    update a live pyqtgraph PlotWidget. woo!
    """
    # initialize main application 
    app = pg.mkQApp()
    pg.setConfigOption('background', 'k')
    pg.setConfigOption('foreground', 'w')
    
    # declare endpoints of interest
    endpoints = ["cage_pressure"]
    start = "2019-10-04T17:30"
    end = "now" 
    
    # declare main window and call whatever
    rp = RabbitPlot()
    
    # connect the RabbitPlot
    pool = QThreadPool()
    listener = RabbitListener()
    listener.signals.target.connect(rp.update_data) # connect to
    pool.start(listener)
    
    # start the main loop
    exit(app.exec_())
    
    
# class RabbitPlot(QWidget):
class RabbitPlot(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('updating plot window')
        self.setGeometry(20, 50, 800, 500)
        
        self.pw = pg.PlotWidget(name='Plot1') 
        self.setCentralWidget(self.pw)
        self.show()
        
        self.p1 = self.pw.plot()
        self.p1.setPen('g')
        self.pw.setLabel('left', "Value", units="V")
        self.pw.setLabel('bottom', "Time", units="s")
        self.p1.setLogMode(xMode=False, yMode=True)
        
        # set the initial values
        # self.yd = np.random.rand(10)
        self.yd = np.zeros(0)
        
    def update_data(self, xv=None, yv=None):
        if yv is not None:
            print("appending:", yv)
            self.yd = np.append(self.yd, yv)

        xd = np.arange(len(self.yd))
        self.p1.setData(y=self.yd, x=xd)


class RabbitSignal(QObject):
    # used by RabbitListener to communicate w/ the main loop (app.exec_())
    # "if you want to define your own signals, they have to be class variables"
    target = pyqtSignal(str, float) 


class RabbitListener(QRunnable):
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
    
        self.channel.queue_bind(exchange=self.config['exchange'],
                                queue=self.config['queue'],
                                routing_key="sensor_value.cage_pressure")

        self.channel.basic_consume(queue=self.config['queue'], 
                                   on_message_callback=self.dispatch, 
                                   auto_ack=True)

        self.channel.start_consuming()


    def dispatch(self, channel, method, properties, body):
        key = method.routing_key
        record = json.loads(body.decode()) # decode binary string to dict
        print('got a record.  name:', key)
        # pprint(record)
        xv = record["timestamp"]
        yv = record["payload"]["value_cal"]
        self.signals.target.emit(xv, yv)





if __name__ == '__main__':
    main()