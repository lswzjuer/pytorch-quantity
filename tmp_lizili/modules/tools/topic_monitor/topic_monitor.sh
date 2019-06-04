#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd ${DIR}

if [ $# -eq 0 ]; then
  python modules/planning/scripts/topic_monitor/topic_monitor.py
else
  if [ $1 == 'loc' ]; then
    rostopic echo /roadstar/localization
  elif [ $1 == 'canbus' ]; then
    rostopic echo /roadstar/canbus/chassis
  elif [ $1 == 'ins' ]; then
    rostopic echo /roadstar/drivers/novatel/ins_stat
  elif [ $1 == 'control' ]; then
    python modules/control/scripts/plot_control/plot_control.py
  else
    python modules/planning/scripts/topic_monitor/topic_monitor.py
  fi
fi
