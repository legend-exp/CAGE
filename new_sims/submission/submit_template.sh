#!/bin/bash

#$ -S /bin/bash #use bash
#$ -m n # don't send mail when job starts or stops.
#$ -w e #verify syntax and give error if so
#$ -V #inherit environment variables
#$ -N g4simpleout #job name
#$ -o /data/eliza1/LEGEND/users/grsong/jobs/logs #standard output of script
#$ -j y 
#$ -l h_rt=10:00:00 #hard time limit, your job is killed if it uses this much cpu.
#$ -l s_rt=9:50:00 #soft time limit, your job gets signaled when you use this much time. Maybe you can gracefully shut down?
#$ -cwd #execute from the current working directory
#$ -t 1-10 #give me N identical jobs, labelled by variable SGE_TASK_ID
#$ -l scratch=10G

#execute the $SGE_TASK_ID'th sub-job
singularity exec --bind /data/eliza1/LEGEND,$TMPDIR /data/eliza1/LEGEND/sw/containers/legend-base.sif /data/eliza1/LEGEND/users/grsong/CAGE/new_sims/submission/run1.sh
