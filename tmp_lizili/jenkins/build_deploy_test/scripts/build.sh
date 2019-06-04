#!/usr/bin/env bash
./docker/scripts/dev_start.sh
./roadstar_docker.sh build_fe
./roadstar_docker.sh build_opt_gpu
if [ $? -eq 0 ]; then
  echo 'roadstar build opt gpu passed!'
else
  exit 1
fi

./roadstar_docker.sh build_driver
if [ $? -eq 0 ]; then
  echo 'roadstar build driver passed!'
else
  exit 2
fi
