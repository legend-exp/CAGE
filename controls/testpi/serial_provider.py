import serial
from serial.tools.list_ports import comports
import threading

from dripline.core import Provider, Spime, calibrate
from dripline.core import exceptions

import logging
logger = logging.getLogger('dragonfly.serial_provider')

class SerialProvider(Provider):

    def __init__(self,
                 port = '',
                 command_terminator='',
                 response_terminator='',
                 serial_kwargs=None,
                 **kwargs):
        Provider.__init__(self, **kwargs)

        self.alock = threading.Lock()
        self.port = port
        self.command_terminator = command_terminator
        self.response_terminator = response_terminator
        self.serial_kwargs = serial_kwargs
        if self.serial_kwargs is None:
            self.serial_kwargs = { 'timeout' : 1 }
        self._reconnect()


    def _reconnect(self):
        logger.info('Establishing serial connection on {}'.format(self.port))
        self.serial = serial.Serial(self.port, **self.serial_kwargs)


    def send(self, commands):
        if isinstance(commands, str):
            commands = [commands]
        self.alock.acquire()

        try:
            data = self._send_commands(commands)
        finally:
            self.alock.release()
        to_return = ";".join(data)
        logger.debug("should return:\n{}".format(to_return))
        return to_return


    def _send_commands(self, commands):
        all_data = []

        for command in commands:
            command += self.command_terminator
            logger.debug("sending: {}".format(repr(command)))
            self.serial.write(command.encode())

            data = self._listen()

            logger.info("sync: {} -> {}".format(repr(command),repr(data)))
            all_data.append(data)
        return all_data


    def _listen(self):
        data = self.serial.read_until(self.response_terminator.encode()).decode(errors='replace')
        if data == '':
            raise exceptions.DriplineHardwareResponselessError("Empty serial.read packet")
        if not data.endswith(self.response_terminator):
            raise exceptions.DriplineHardwareError("Unexpected serial.timeout, received: {}".format(data))
        logger.debug(repr(data))
        return data[0:data.rfind(self.response_terminator)]



class USBSerialProvider(SerialProvider):

    def __init__(self,
                 manufacturer,
                 **kwargs):

        self.manufacturer = manufacturer
        
        SerialProvider.__init__(self, port=self.find_usb(), **kwargs)


    def find_usb(self):
        ports = [x.device for x in comports() if 'USB' in x.device and self.manufacturer == x.manufacturer]
        if len(ports) != 1:
            raise exceptions.DriplineHardwareConnectionError('Invalid USB port: expected 1 and found {} - {}'.format(len(ports),ports))

        return ports[0]



class LNGetSpime(Spime):

    def __init__(self, get_str, **kwargs):
        Spime.__init__(self, **kwargs)
        self._get_str = get_str

    @calibrate()
    def on_get(self):
        reply = self.provider.send(self._get_str)
        # reply format will be "<sign>XXX.XX kg" - need to split off the unit and strip out any spaces to parse
        return float(reply.split(' kg')[0].replace(' ',''))
        
    def on_set(self, value):
        raise exceptions.DriplineMethodNotSupportedError('setting not available for {}'.format(self.name))



def status_calibration(value):
    caldict = { 0 : "On",
                1 : "Ramp Up",
                2 : "Ramp Down",
                3 : "Over Current",
                4 : "Over Voltage",
                5 : "Under Voltage",
                6 : "MaxV Protection",
                7 : "Off via Trip",
                8 : "Over Power",
                9 : "Over Temperature",
                10 : "Disabled",
                11 : "Killed",
                12 : "Interlocked",
                13 : "Calibration Error" }
    status = []
    for i in range(14):
        if value & 1<<i:
            status.append(caldict[i])
    if len(status) == 0:
        return "Off"
    return ";".join(status)



class CAENHVSpime(Spime):

    def __init__(self, base_str, **kwargs):
        Spime.__init__(self, **kwargs)
        self._get_str = base_str.format("MON")
        self._set_str = base_str.format("SET") + ",VAL:{}"

    @calibrate([status_calibration])
    def on_get(self):
        reply = self.provider.send(self._get_str)
        self.parse_reply(reply, "get")
        reply = reply.split("VAL:")[1].lstrip('0')
        if reply == '' or reply == '.0':
            reply = '0'
        return reply

    def on_set(self, value):
        reply = self.provider.send(self._set_str.format(value))
        self.parse_reply(reply, "set")
        return reply

    def parse_reply(self, reply, comm):
        if reply[1:6] != self._get_str[1:6]:
            raise exceptions.DriplineHardwareError("Badly formatted response to {} {}: {}".format(self.name, comm, reply))
        elif reply.endswith("ERR"):
            raise exceptions.DriplineHardwareError("Error code returned from {} {}: {}".format(self.name, comm, reply))
        elif reply[7:13] != "CMD:OK":
            raise exceptions.DriplineHardwareError("Unexpected response to {} {}: {}".format(self.name, comm, reply))



class CAENHVFormatSpime(CAENHVSpime):

    def __init__(self,
                 get_str,
                 set_str,
                 set_value_map,
                 **kwargs):
        CAENHVSpime.__init__(self, base_str='', **kwargs)
        self._get_str = get_str
        self._set_str = set_str
        self._set_value_map = set_value_map
        
    def on_set(self, value):
        if value in self._set_value_map:
            logger.debug("mapping value {} to {}".format(value, self._set_value_map[value]))
            value = self._set_value_map[value]
        reply = self.provider.send(self._set_str.format(value))
        self.parse_reply(reply, "set")
        return reply



class CAENHVGetSpime(CAENHVSpime):

    def __init__(self, **kwargs):
        CAENHVSpime.__init__(self, **kwargs)

    def on_set(self, value):
        raise exceptions.DriplineMethodNotSupportedError('setting not available for {}'.format(self.name))
