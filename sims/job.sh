#!/bin/bash

## Usage: qsub muon_job.sh
##
#$ -S /bin/bash #use bash
#$ -w e #verify syntax and give error if so
#$ -V #inherit environment variables
#$ -N job #job name
#$ -e ~/scanner #error output of script
#$ -o ~/scanner #standard output of script
#$ -l scratch=2G
#$ -l h_rt=10:00:00 #hard time limit, your job is killed if it uses this much cpu.
#$ -l s_rt=9:50:00 #soft time limit, your job gets signaled when you use this much time. Maybe you can gracefully shut down?
#$ -cwd #execute from the current working directory
#$ -t 1-1 #give me 128 identical jobs, labelled by variable SGE_TASK_ID

## the job

g4simple muonsourcerun.mac
