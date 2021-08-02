#!/bin/bash
#SBATCH --qos=shared
#SBATCH --time=48:00:00
#SBATCH --constraint=haswell
#SBATCH --account=m2676
#SBATCH --export=HDF5_USE_FILE_LOCKING=FALSE
#SBATCH --image=legendexp/legend-base:latest
#SBATCH --chdir=/global/project/projectdirs/legend/software/CAGE/processing
#SBATCH --output=/global/project/projectdirs/legend/software/CAGE/processing/logs/cori-%j.txt
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=wisecg@uw.edu

echo "Job Start:"
date
echo "Node(s):  "$SLURM_JOB_NODELIST
echo "Job ID:  "$SLURM_JOB_ID

if [ -n "$SHIFTER_RUNTIME" ]; then
  echo "Shifter image active."
  echo "pwd: "`pwd`
  echo "gcc: "$CC
  echo "g++:"$CXX
  echo "Python:"`python --version`
  echo "ROOT:"`root-config --version`
fi

# update fileDB (usually want to run this first)
# NOTE: you need to update runDB.json before running this!
# shifter python setup.py --update --orca -b

# -- Campaign 2 workspace --
shifter python processing.py -q 'run >= 297 and run <= 304' --d2r
# shifter python setup.py --orca --rt -b


# -- Campaign 1 workspace --

# run dsp_to_hit for Campaign 1
# shifter python processing.py -q 'dsp_id==1 or dsp_id==2' --d2h -o

# reprocess specific dsp_id's.  roughly 5 min/cycle file.
#shifter python processing.py -q 'dsp_id==1' --r2d -o
#shifter python processing.py -q 'dsp_id==2' --r2d -o


# -- reprocess 2021 d2r (ts bug)
# shifter python processing.py -q 'cycle>=1192' --d2r -o
# not re-r2d'ing cycles before 1192 because config_dsp would be wrong for them ...
#shifter python processing.py -q 'cycle>=1192' --r2d -o

# --
# Standard mode: update recent runs (cuts down on log file size)
# shifter python processing.py -q 'run>=118' --d2r --r2d

# update everything (no overwriting)
# shifter python processing.py -q 'cycle>0' --d2r --r2d

# overwrite everything (>> 24 hr job)
# shifter python processing.py -q 'cycle>0' --d2r --r2d -o

# SPECIAL: d2r new runs and rerun r2d with a new processor list (25 GB/hr)
# shifter python processing.py -q 'cycle>=560' --r2d -o

# This runs whatever we pass to it (maybe from python)
# echo "${@}"
# ${@}

echo "Job Complete:"
date
