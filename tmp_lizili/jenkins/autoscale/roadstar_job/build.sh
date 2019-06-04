#!/usr/bin/env bash

cd /roadstar

./roadstar.sh build_fe
if [ $? -eq 0 ]; then
  echo 'roadstar build fe passed!'
else
  echo 'roadstar build fe failed!'
  exit 1
fi
./roadstar.sh build_opt_gpu
if [ $? -eq 0 ]; then
  echo 'roadstar build opt gpu passed!'
else
  echo 'roadstar build opt gpu failed!'
  exit 2
fi


