# The CAGE motor controls system

## Installation Notes


### [GalilTools (with Python support)](http://www.galilmc.com/sw/pub/all/doc/gclib/html/python.html)

The install instructions listed on the page will not work for a Python3 system.  Please follow these instructions instead.  First set the environment variable in your `.bashrc` or `envSetup.sh` file:
```
export DYLD_LIBRARY_PATH="/Applications/gclib/dylib/:${DYLD_LIBRARY_PATH}"
```
Then, do the following:
```
mkdir ~/software/gclib
cd ~/software/gclib
cp /Applications/gclib/source/gclib_python.tar.gz .
tar -xvf gclib_python.tar.gz
cd ..
pip install -e gclib
```
To test if the installation worked, navigate to a **different folder** than the one containing `gclib.py` and try the following:
```
$ python (should be aliased to python3)

Python 3.7.3 (default, Mar 27 2019, 09:23:15)
>>> import gclib
>>> g = gclib.py()
>>> g.GOpen('[IP_address_here] --direct')
>>> print(g.GInfo())
[IP_address_here], DMC2142sH2a, 29903
```