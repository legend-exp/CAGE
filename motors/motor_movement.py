#!/usr/bin/env python3
import argparse

def main():
    """
    to-do routines:
    - limit_switch_check
    """
    par = argparse.ArgumentParser(description="a program that does a thing")
    arg, st, sf = par.add_argument, "store_true", "store_false"
    # arg("-a", "--aaa", action=st, help="take a flag")
    # arg("-b", "--bbb", nargs=1, help="take a flag with argument")
    # arg("-c", "--ccc", nargs='*', action="store", help="variable num. args")
    args = vars(par.parse_args())
    
    config = {
        'source' = {'rpi_pin':11},
        'linear' = {'rpi_pin':11},
        'rotary' = {'rpi_pin':11},
        }

    
    

    
    
    
if __name__=="__main__":
    main()