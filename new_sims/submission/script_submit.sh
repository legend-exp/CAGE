#!/bin/bash

rad=(5 6 7 8 9 10);
det=(90);
rot=(162);

for y in ${rad[@]}; do
    for d in ${det[@]}; do
       for t in ${rot[@]}; do
          qsub "sub_y${y}_thetaDet${d}_rot${t}.sh";
       done
    done
done
