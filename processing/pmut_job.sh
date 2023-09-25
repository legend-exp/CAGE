#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -c 1
#SBATCH -q shared
#SBATCH -J cage_job
#SBATCH -t 12:00:00
#SBATCH -A m2676
#SBATCH --export=HDF5_USE_FILE_LOCKING=FALSE
#SBATCH --image=legendexp/legend-base:latest
#SBATCH --chdir=/global/cfs/cdirs/legend/software/CAGE/processing
#SBATCH --output=/global/cfs/cdirs/legend/software/CAGE/processing/logs/pmut-%j.txt
#SBATCH --mail-type=ALL
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

shifter python processing.py -q 'run>=516' --d2r
# shifter python3 setup.py --rebuild

# shifter python processing.py -q 'run>=516' --d2r --r2d
# shifter python3 setup.py --rebuild

# This runs whatever we pass to it (maybe from python)
# echo "${@}"
# ${@}

echo "Job Complete:"
date
