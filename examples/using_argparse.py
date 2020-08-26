#!/usr/bin/env python3
import argparse
from pprint import pprint

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
    arg("-a", "--aaa", action=st, help="take a flag")
    arg("-b", "--bbb", nargs=1, help="take a flag with argument")
    arg("-c", "--ccc", nargs='*', action="store", help="take a variable num. args")
    args = vars(par.parse_args())
    
    # now parse the arguments like a dictionary of bools
    if args["aaa"]:
        func1()
        
    if args["bbb"]:
        mode = int(args["bbb"][0]) if args["bbb"] else 0
        func2(mode)
        
    if args["ccc"]:
        thing1, thing2 = args["ccc"][0], args["ccc"][1]
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
    
    var1, var2, var3 = 1, 2, 3
    # print("var1", var1, " var2", var2, " var3")
    print(f"var1 {var1}  var2 {var2}  var3 {var3}")


def func3(thing1, thing2):
    """
    docstrings can help auto-generate whole documentation webpages, use them!
    """
    print("Life is pain. Anyone who says differently is selling something.")
    print(f"Oh also, {thing1} and {thing2}")
    
    a = 1.23423523452345
    
    plt.plot(a, b, label=f"{a:.2f}")
    
    str1 = "{:.2f} {:.3f}".format(a, b)
    
    str2 = f"{a:.2f} {b:.3f}")
    

    
if __name__=="__main__":
    main()
    