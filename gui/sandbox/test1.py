import json
import traceback
import pika
import sys

from pprint import pprint

from PyQt5.QtWidgets import QApplication, QLabel
from PyQt5.QtCore import QThreadPool, QObject, QRunnable, pyqtSignal

def main():
    app = QApplication([])
    widget = MainWidget()
    pool = QThreadPool()
    listener = MessageListener()
    listener.signals.target.connect(widget.show)
    pool.start(listener)
    app.exec_()

class RabbitMQSignals(QObject):
    target = pyqtSignal(int)

class MessageListener(QRunnable):

    def __init__(self):
        super(MessageListener, self).__init__()
        self.signals = RabbitMQSignals()

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
                                routing_key="sensor_value.#")
        
        self.channel.basic_consume(queue=self.config['queue'], 
                                   on_message_callback=self.dispatch, 
                                   auto_ack=True)

        print("wabbit eatin hay")
        self.channel.start_consuming()


    def dispatch(self, channel, method, properties, body):
        
        key = method.routing_key
        record = json.loads(body.decode()) # decode binary string to dict
        print('got a record.  name:', key)
        pprint(record)
        
        if body == 'quit':
            sys.exit(0)

        self.signals.target.emit(0)


class MainWidget(QObject):

    def __init__(self):
        super().__init__()
        self.label = QLabel("Hello World")

    def show(self, action):
        print('[x] Dispatched :' + str(action))
        self.label.show()


if __name__ == '__main__':
    main()