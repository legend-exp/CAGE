#!/usr/bin/env python3
"""
"""
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from source_move_beta import *
from linear_move_beta import *
from rotary_move_beta import *
from motor_movement_beta import movement_program


def main():
    """
    convention: let's try using the D prefix on our objects
    """
    app = QApplication(sys.argv)
    ex = DCAGE()
    sys.exit(app.exec_())


class DCAGE(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('DCAGE: Data acquisition for CAGE')
        self.setGeometry(0, 0, 800, 600)
        self.ctr_widget = DTabWindow(self)
        self.setCentralWidget(self.ctr_widget)
        self.show()


class DTabWindow(QWidget):

    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.layout = QVBoxLayout(self)

        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tabs.resize(300,200)

        # Add tabs
        self.tabs.addTab(self.tab1,"Motors")
        self.tabs.addTab(self.tab2,"Slow Controls")

        # Create tabs
        self.tab1.layout = QVBoxLayout(self)
        self.tab2.layout = QVBoxLayout(self)
        self.pushButton1 = QPushButton("Move Motors...or dont, i dont care")
        self.tab1.layout.addWidget(self.pushButton1)
        self.tab1.setLayout(self.tab1.layout)
        self.pushButton2 = QPushButton('Slow Controls Monitoring')
        self.tab2.layout.addWidget(self.pushButton2)
        self.tab2.setLayout(self.tab2.layout)

        # Add tabs to widget
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

        self.pushButton1.clicked.connect(self.on_motor_clicked)
        # self.dialog = DMotors(self)

        self.pushButton2.clicked.connect(self.on_slowControls_clicked)
        # self.dialog = DSlowControls(self)


    def on_motor_clicked(self):
        self.dialog = DMotors(self)
        self.dialog.show()
    def on_slowControls_clicked(self):
        self.dialog = DSlowControls(self)
        self.dialog.show()

class DMotors(QMainWindow):
    def __init__(self, parent=None):
        super(DMotors, self).__init__(parent)
        self.ctr_widget = DMotorWindow(self)
        self.setCentralWidget(self.ctr_widget)


class DMotorWindow(QWidget):
    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.layout = QHBoxLayout(self)


        self.pushButton1 = QPushButton('Movement Program')
        self.pushButton2 = QPushButton('Linear Stage')
        self.pushButton3 = QPushButton('Source Motor')


        self.layout.addWidget(self.pushButton1)
        self.layout.addWidget(self.pushButton2)
        self.layout.addWidget(self.pushButton3)
        self.setLayout(self.layout)

        self.pushButton1.clicked.connect(self.movement_click)
        self.pushButton2.clicked.connect(self.linear_click)
        self.pushButton3.clicked.connect(self.source_click)

    @pyqtSlot()
    def movement_click(self):
        movement_program()
    def linear_click(self):
        linear_program()
    def source_click(self):
        source_program()

class DSlowControls(QMainWindow):
    def __init__(self, parent=None):
        super(DSlowControls, self).__init__(parent)
        # self.ctr_widget = DMotorWindow(self)
        # self.setCentralWidget(self.ctr_widget)

# @pyqtSlot()
    # for currentQTableWidgetItem in self.ctr_widget.selectedItems():
    #     print(currentQTableWidgetItem.row(), currentQTableWidgetItem.column(), currentQTableWidgetItem.text())



if __name__=="__main__":
    main()
