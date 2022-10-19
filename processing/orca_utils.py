import sys, gzip
import numpy as np
import plistlib
import pprint as pp


def open_orca(orca_filename):
    if orca_filename.endswith('.gz'):
        return gzip.open(orca_filename.encode('utf-8'), 'rb')
    else: return open(orca_filename.encode('utf-8'), 'rb')

def from_bytes(data, big_endian=False):
    """
    python2 doesn't have this function,
    so rewrite it for backwards compatibility
    """
    if isinstance(data, str):
        data = bytearray(data)
    if big_endian:
        data = reversed(data)
    num = 0
    for offset, byte in enumerate(data):
        num += byte << (offset * 8)
    return num


def parse_header(orca_filename):
    """
    Opens the given file for binary read ('rb'), then grabs the first 8 bytes
    The first 4 bytes (1 long) of an orca data file are the total length in
    longs of the record
    The next 4 bytes (1 long) is the length of the header in bytes
    The header is then read in ...
    """
    with open_orca(orca_filename) as xmlfile_handle:
        #read the first word:
        ba = bytearray(xmlfile_handle.read(8))

        #Replacing this to be python2 friendly
        # #first 4 bytes: header length in long words
        # i = int.from_bytes(ba[:4], byteorder=sys.byteorder)
        # #second 4 bytes: header length in bytes
        # j = int.from_bytes(ba[4:], byteorder=sys.byteorder)

        big_endian = False if sys.byteorder == "little" else True
        i = from_bytes(ba[:4], big_endian=big_endian)
        j = from_bytes(ba[4:], big_endian=big_endian)
        if (np.ceil(j/4) != i-2) and (np.ceil(j/4) != i-3):
            print('Error: header byte length = %d is the wrong size to fit into %d header packet words' % (j, i-2))
            return i, j, {}

        #read in the next that-many bytes that occupy the plist header
        as_bytes = xmlfile_handle.read(j)
        ba = bytearray(as_bytes)

        #convert to string
        #the readPlistFromBytes method doesn't exist in 2.7
        if sys.version_info[0] < 3:
            header_string = ba.decode("utf-8")
            header_dict = plistlib.readPlistFromString(header_string)
        elif sys.version_info[1] < 9:
            header_dict = plistlib.readPlistFromBytes(ba)
        else:
            header_dict = plistlib.loads(as_bytes, fmt=plistlib.FMT_XML)
        return i, j, header_dict
    
# Not an ORCA function but old pygama functionality that we use in CAGE processing
def write_pretty(db_dict, f_db):
    """
    write a TinyDB database to a special pretty-printed JSON file.
    if i cared less about the format, i would just call TinyDB(index=2)
    """
    pretty_db = {}
    for tb_name, tb_vals in db_dict.items(): # convert pprint integer keys to str
        pretty_db[tb_name] = {str(tb_idx):tb_row for tb_idx, tb_row in tb_vals.items()}
    pretty_str = pp.pformat(pretty_db, indent=2, width=120) #sort_dicts is Py>3.8
    pretty_str = pretty_str.replace("'", "\"")
    with open(f_db, 'w') as f:
        f.write(pretty_str)