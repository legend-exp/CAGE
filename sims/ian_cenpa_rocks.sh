#!/bin/bash

## Usage: qsub SimulateM1Co56.sh
##
#$ -S /bin/bash #use bash
#$ -m n # don't send mail when job starts or stops.
#$ -w e #verify syntax and give error if so
#$ -V #inherit environment variables
#$ -N mage_M1Co56 #job name
#$ -e ~/logs #error output of script
#$ -o ~/logs #standard output of script
#$ -l scratch=2G
#$ -l h_rt=10:00:00 #hard time limit, your job is killed if it uses this much cpu.
#$ -l s_rt=9:50:00 #soft time limit, your job gets signaled when you use this much time. Maybe you can gracefully shut down?
#$ -cwd #execute from the current working directory
#$ -t 1-1000 #give me 128 identical jobs, labelled by variable SGE_TASK_ID

## the job

cd $TMPDIR

NUMEVENTS=1000000
MACRO=~/macros/Co56M1LineSrc_template.mac
DETCONFIG=~/macros/det_config_DS6.json
OUTPUTDIR=$MYLEGENDDIR/M1Co56Sims
JOBNUM=$((JOB_ID * 1000 + SGE_TASK_ID))

echo sed -e "s/RUNID/${JOBNUM}/g" -e "s/NEVENTS/${NUMEVENTS}/g" ${MACRO} > macro.mac
sed -e "s/RUNID/${JOBNUM}/g" -e "s/NEVENTS/${NUMEVENTS}/g" ${MACRO} > macro.mac
echo cp ${DETCONFIG} detconfig.json
cp ${DETCONFIG} detconfig.json

echo
echo MaGe macro.mac
MaGe macro.mac

echo
echo process_MJD_as_built_mage_results -c detconfig.json *.root
process_MJD_as_built_mage_results -c detconfig.json *.root

echo
echo ls -l
ls -l

echo
echo cp *.root ${OUTPUTDIR}
cp *.root ${OUTPUTDIR}
