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
#shifter python processing.py -q 'run == 516' -o --r2d --dsp dsp_cyc4430.json
#shifter python processing.py -q 'run == 517' -o --r2d --dsp dsp_cyc4442.json
#shifter python processing.py -q 'run == 518' -o --r2d --dsp dsp_cyc4455.json
#shifter python processing.py -q 'run == 519' -o --r2d --dsp dsp_cyc4467.json
#shifter python processing.py -q 'run == 520' -o --r2d --dsp dsp_cyc4489.json
#shifter python processing.py -q 'run == 521' -o --r2d --dsp dsp_cyc4508.json
#shifter python processing.py -q 'run == 522' -o --r2d --dsp dsp_cyc4520.json
#shifter python processing.py -q 'run == 523' -o --r2d --dsp dsp_cyc4531.json
#shifter python processing.py -q 'run == 524' -o --r2d --dsp dsp_cyc4543.json
#shifter python processing.py -q 'run == 525' -o --r2d --dsp dsp_cyc4555.json
#shifter python processing.py -q 'run == 526' -o --r2d --dsp dsp_cyc4566.json
#shifter python processing.py -q 'run == 527' -o --r2d --dsp dsp_cyc4578.json
#shifter python processing.py -q 'run == 528' -o --r2d --dsp dsp_cyc4590.json
#shifter python processing.py -q 'run == 529' -o --r2d --dsp dsp_cyc4604.json
#shifter python processing.py -q 'run == 530' -o --r2d --dsp dsp_cyc4616.json
#shifter python processing.py -q 'run == 531' -o --r2d --dsp dsp_cyc4628.json
#shifter python processing.py -q 'run == 532' -o --r2d --dsp dsp_cyc4640.json
#shifter python processing.py -q 'run == 533' -o --r2d --dsp dsp_cyc4652.json
shifter python processing.py -q 'run == 534' -o --r2d --dsp dsp_cyc4688.json


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
