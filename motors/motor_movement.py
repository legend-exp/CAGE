#!/usr/bin/env python3
import argparse
import gclib

def main():
    """
    REFERENCE:
    https://elog.legend-exp.org/UWScanner/200130_140616/cage_electronics.pdf
    
    TODO functions:
    - limit_switch_check
    """
    par = argparse.ArgumentParser(description="a program that does a thing")
    arg, st, sf = par.add_argument, "store_true", "store_false"
    # arg("-a", "--aaa", action=st, help="take a flag")
    # arg("-b", "--bbb", nargs=1, help="take a flag with argument")
    # arg("-c", "--ccc", nargs='*', action="store", help="variable num. args")
    args = vars(par.parse_args())
    
    config = {
        'controller': {'ip':'172.25.100.168', 'model':'DMC2142sH2a'},
        'source': {'rpi_pin':13, 'axis':'A'},
        'linear': {'rpi_pin':7, 'axis':'B'},
        'rotary': {'rpi_pin':11, 'axis':'D'},
        }
    
    # handy globals (used by almost all routines)
    global gcpy, gcmd
    gcpy = gclib.py()
    gcmd = gcpy.GCommand

    # run routines.  TODO: run via argparse
    test_readout()
    
    

def test_readout():
    """ 
    Verify that we're connected to the Newmark motor controller.
    """
    g = gclib.py()
    c = g.GCommand
    g.GOpen('172.25.100.168 --direct')
    print(type(g.GInfo()))
    print(g.GAddresses())
    print(g.GVersion())
    print(g.GInfo())



    
    
    
if __name__=="__main__":
    main()