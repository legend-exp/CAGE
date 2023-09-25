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
#SBATCH --chdir=/global/cfs/projectdirs/legend/software/CAGE/processing
#SBATCH --output=/global/cfs/projectdirs/legend/software/CAGE/processing/logs/pmut-%j.txt
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=grsong@uw.edu

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

# shifter python processing.py -q 'run >= 428' --d2r

# shifter python processing.py -q 'run >= 395' --d2r

# shifter python processing.py -q 'run >= 385 and run <= 391' --d2r --r2d --mc -o
# shifter python setup.py --orca --rt -b

#shifter python processing.py -q 'run >= 358 and run <= 383' --d2h
# shifter python processing.py -q 'run >= 392' --r2d -o
#shifter python processing.py -q 'run == 384' --d2h

#shifter python processing.py -q 'run > 305 and run < 332' --d2h
#shifter python processing.py -q 'run >= 355 and run <= 357' --r2d -o
#shifter python processing.py -q 'run >= 358 and run <= 383' --r2d -o
#shifter python processing.py -q 'run == 384' --r2d -o


# run a bunch of DSP in parallel (lazy method, comment each line in & submit)
shifter python processing.py -q 'run == 493' -o --r2d --dsp dsp_cyc4060.json
#shifter python processing.py -q 'run == 494' -o --r2d --dsp dsp_cyc4065.json
#shifter python processing.py -q 'run == 495' -o --r2d --dsp dsp_cyc4097.json
#shifter python processing.py -q 'run == 496' -o --r2d --dsp dsp_cyc4105.json
#shifter python processing.py -q 'run == 497' -o --r2d --dsp dsp_cyc4117.json
#shifter python processing.py -q 'run == 498' -o --r2d --dsp dsp_cyc4129.json
#shifter python processing.py -q 'run == 499' -o --r2d --dsp dsp_cyc4142.json
#shifter python processing.py -q 'run == 500' -o --r2d --dsp dsp_cyc4154.json
#shifter python processing.py -q 'run == 501' -o --r2d --dsp dsp_cyc4166.json
#shifter python processing.py -q 'run == 502' -o --r2d --dsp dsp_cyc4178.json
#shifter python processing.py -q 'run == 503' -o --r2d --dsp dsp_cyc4190.json
#shifter python processing.py -q 'run == 504' -o --r2d --dsp dsp_cyc4202.json
#shifter python processing.py -q 'run == 505' -o --r2d --dsp dsp_cyc4213.json
#shifter python processing.py -q 'run == 506' -o --r2d --dsp dsp_cyc4224.json
#shifter python processing.py -q 'run == 507' -o --r2d --dsp dsp_cyc4236.json

#shifter python processing.py -q 'run == 516' -o --r2d --dsp dsp_cyc4430.json
#shifter python processing.py -q 'run == 517' -o --r2d --dsp dsp_cyc4442.json
#shifter python processing.py -q 'run == 518' -o --r2d --dsp dsp_cyc4455.json

#shifter python processing.py -q 'cycle >= 3819 and run <= 471' -o --r2d --dsp dsp_cyc3717.json
#shifter python processing.py -q 'run == 430' --r2d --dsp dsp_cyc3121.json
#shifter python processing.py -q 'run == 320' --r2d
#shifter python processing.py -q 'run >= 321 and run <= 331' --d2r --r2d -o
#shifter python processing.py -q 'run == 332' --d2r --r2d
#shifter python processing.py -q 'run >= 333 and run <= 340' --d2r --r2d
#shifter python processing.py -q 'run == 341' --d2r --r2d
#shifter python processing.py -q 'run >= 342 and run <= 346' --d2r --r2d
#shifter python processing.py -q 'run >= 347 and run <= 353' --d2r --r2d

# shifter python processing.py -q 'run >= 297 and run <= 304' --d2r
# shifter python setup.py --orca --rt -b

# energy cal command (doesn't really need a whole job, but for posterity this is what i did)
# shifter python energy_cal.py -q 'run>=238 and run <=303' -b -p --all

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
