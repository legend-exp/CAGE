#!/bin/bash

rad=(14);
det=(60 75 90);
rot=(0 180 145);

for y in ${rad[@]}; do
    for d in ${det[@]}; do
       for t in ${rot[@]}; do
          qsub "sub_y${y}_thetaDet${d}_rot${t}.sh";
       done
    done
done
