import board, busio, digitalio
import adafruit_max31865
from dripline.core import Spime, calibrate
from dripline.core.exceptions import DriplineMethodNotSupportedError

class Max31865Spime(Spime):

    def __init__(self,
                 cspin,
                 nwires,
                 ref_resistor=430.0,
                 **kwargs):
        Spime.__init__(self, **kwargs)

        spi = busio.SPI(board.SCK, MOSI=board.MOSI, MISO=board.MISO)
        cs = digitalio.DigitalInOut(board.pin.Pin(cspin))
        self.sensor = adafruit_max31865.MAX31865(spi, cs, ref_resistor=ref_resistor, wires=nwires)

    @calibrate()
    def on_get(self):
        return self.sensor.temperature

    def on_set(self):
        raise DriplineMethodNotSupportedError('setting not available for {}'.format(self.name))

