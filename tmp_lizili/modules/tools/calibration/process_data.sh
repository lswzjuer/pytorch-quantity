#!/usr/bin/env bash



#! /bin/bash

# set -x

rm result.csv
for f in `ls ${1}/*_recorded.csv`
do
    echo "Processing $f"
    python -W ignore process_data.py $f
done
