#!/bin/bash
#SBATCH --qos=shared
#SBATCH --time=10:00:00
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

# update recent runs (cuts down on log file size)
shifter python processing.py -q 'run>95' --d2r --r2d

# update everything (no overwriting)
# shifter python processing.py -q 'cycle>0' --d2r --r2d

# overwrite everything (~24 hr job)
# shifter python processing.py -q 'cycle>0' --d2r --r2d -o

# SPECIAL: d2r new runs and rerun r2d with a new processor list (25 GB/hr)
# shifter python processing.py -q 'cycle>=560' --r2d -o

# This runs whatever we pass to it (maybe from python)
# echo "${@}"
# ${@}

echo "Job Complete:"
date
