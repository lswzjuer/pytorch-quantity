#!/usr/bin/env bash

MODULE_NAME=control
if [ $# -eq 1 ]; then
  MODULE_NAME=$1
fi
PID=`ps aux | grep /roadstar/bazel-bin/modules/$MODULE_NAME/$MODULE_NAME | grep -v grep | awk '{print $2}'`
if [ ! $PID ]; then
  echo "No process: "$MODULE_NAME
else
  echo $MODULE_NAME" pid is "$PID
  sudo renice -19 -p $PID
fi
