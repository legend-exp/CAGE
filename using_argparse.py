#!/usr/bin/env python3
import argparse

def main():
    """
    A basic example of using argparse in a python program to make it
    usable by command line.  i've found this very useful to get flexible
    programs working quickly. 
    
    Another trick I often use is making a program executable, by typing:
        chmod a+x using_argparse.py
    Then I can run it with less typing:
        ./using_argparse.py [args]
    """
    
    # I typically paste a block like this into a program's main function
    par = argparse.ArgumentParser(description="a program that does a thing")
    arg, st, sf = par.add_argument, "store_true", "store_false"
    arg("-a", "--ccc", action=st, help="take a bool flag")
    arg("-b", "--bbb", nargs=1, help="take a bool flag with argument")
    arg("-c", nargs='*', action="store", help="take a variable num. args")
    args = vars(par.parse_args())

    # now parse the arguments like a dictionary of bools
    if args["a"]:
        func1()
        
    if args["b"]:
        mode = int(args["b"][0]) if args["b"] else 0
        func2(mode)
        
    if args["c"]:
        thing1, thing2 = args["c"][0], args["c"][1]
        func3(thing1, thing2)


# here are some placeholder functions

def func1():
    """
    almost all functions should have a docstring below the "def" like this.
    """
    print("Make it so!")


def func2(mode):
    """
    don't forget the docstring!
    """
    print("Engage!")
    print(f"Oh also, here's a cool way to format strings: {mode}")


def func3(thing1, thing2):
    """
    docstrings can help auto-generate whole documentation webpages, use them!
    """
    print("Life is pain. Anyone who says differently is selling something.")
    print(f"Oh also, {thing1} and {thing2}")

    
if __name__=="__main__":
    main()
    