# The CAGE motor controls system

## Installation Notes


### [GalilTools (with Python support)](http://www.galilmc.com/sw/pub/all/doc/gclib/html/python.html)

**Note:** The `setup.py` script contained in `gclib_python.tar.gz` will appear to install to a python2-specific folder:
```
running install_egg_info
Removing /usr/local/lib/python2.7/site-packages/gclib-1.0-py2.7.egg-info
Writing /usr/local/lib/python2.7/site-packages/gclib-1.0-py2.7.egg-info
```
However, it seems that it still works on a Python3 system, provided the environment variable is set correctly in `.bashrc` or `envSetup.sh`:
```
export DYLD_LIBRARY_PATH="/Applications/gclib/dylib/:${DYLD_LIBRARY_PATH}"
```
```
Python 3.7.3 (default, Mar 27 2019, 09:23:15)
>>> import gclib
>>> g = gclib.py()
>>> g.GOpen('[IP_address_here] --direct')
>>> print(g.GInfo())
[IP_address_here], DMC2142sH2a, 29903
```