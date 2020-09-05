
## Data Processing

CAGE processing is organized using the `DataGroup` class from pygama.  

A user must scan the DAQ directory for new files, and create a `fileDB` with a row for every file.

This is handled by the `setup.py` routine.  It is smart enough to be able to update an existing `fileDB` as new data comes in.  

So the rough procedure for running it is:

First time setup:
```
./setup.py --mkdirs  (--lh5_user is optional)

./setup.py --init  (runs initial scan of DAQ directory)
```

Normal operations:
```
./setup.py --update  (scans DAQ directory for new files)

./setup.py --orca (scans ORCA headers for UNIX start time & threshold)

(run daq_to_raw and raw_to_dsp processing)

./setup.py --rt (scans DSP files to get runtimes, needed by energy_cal)
```
