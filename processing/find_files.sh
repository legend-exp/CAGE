#!/bin/bash
for f in ${CAGE_DAQ}/2022/10/10/Data/*
do
    name="$(basename -- $f)"
    if ! test -f ${CAGE_DAQ}/2022/10/Data/$name ; then
        echo $name
    fi
done
