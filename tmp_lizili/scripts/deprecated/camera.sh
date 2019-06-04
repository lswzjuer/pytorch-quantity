#!/usr/bin/env bash

###############################################################################
# Copyright 2017 The Roadstar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "${DIR}/.."

source "${DIR}/roadstar_base.sh"

DATE=$(date +%F)
ROS_LOG_DIR="${ROADSTAR_ROOT_DIR}/data/log/${DATE}/camera"
LOG="${ROADSTAR_ROOT_DIR}/data/log/${DATE}/camera.log"
# ${HOME} is required no matter what ROS_LOG_DIR is. BUT is is created only if ROS_LOG_DIR is not set...
mkdir -p ${HOME}/.ros

if [ ! -e "${ROS_LOG_DIR}" ]; then
  mkdir -p "${ROS_LOG_DIR}"
fi

function start() {
  launch_file=""
  if [[ "$@" = *"raw"* ]]; then
    launch_file="run_camera"
  else
    launch_file="run_camera_and_compression"
  fi

  CMD="roslaunch pylon_camera ${launch_file}.launch"
  NUM_PROCESSES="$(pgrep -c -f "${launch_file}")"
  if [ "${NUM_PROCESSES}" -eq 0 ]; then
    eval "ROS_LOG_DIR=${ROS_LOG_DIR} nohup ${CMD} </dev/null >${LOG} 2>&1 &"
  fi
}

function start_fe() {
  launch_file=""
  if [[ "$@" = *"raw"* ]]; then
    launch_file="run_camera"
  else
    launch_file="run_camera_and_compression"
  fi

  CMD="ROS_LOG_DIR=${ROS_LOG_DIR} roslaunch pylon_camera ${launch_file}.launch"
  NUM_PROCESSES="$(pgrep -c -f "${launch_file}")"
  if [ "${NUM_PROCESSES}" -eq 0 ]; then
    eval "${CMD}"
  fi
}


function start_with() {
  param=""
  if [[ "$@" = *"head_left"* ]]; then
    param="${param} head_left:=true"
  else
    param="${param} head_left:=false"
  fi

  if [[ "$@" = *"head_right"* ]]; then
    param="${param} head_right:=true"
  else
    param="${param} head_right:=false"
  fi

  if [[ "$@" = *"mid_left"* ]]; then
    param="${param} mid_left:=true"
  else
    param="${param} mid_left:=false"
  fi

  if [[ "$@" = *"mid_right"* ]]; then
    param="${param} mid_right:=true"
  else
    param="${param} mid_right:=false"
  fi

  if [[ "$@" = *"front_left"* ]]; then
    param="${param} front_left:=true"
  else
    param="${param} front_left:=false"
  fi
  if [[ "$@" = *"front_right"* ]]; then
    param="${param} front_right:=true"
  else
    param="${param} front_right:=false"
  fi

  if [[ "$@" = *"tail_left"* ]]; then
    param="${param} tail_left:=true"
  else
    param="${param} tail_left:=false"
  fi
  if [[ "$@" = *"tail_right"* ]]; then
    param="${param} tail_right:=true"
  else
    param="${param} tail_right:=false"
  fi

  launch_file=""
  if [[ "$@" = *"raw"* ]]; then
    launch_file="run_camera"
  else
    launch_file="run_camera_and_compression"
  fi

  CMD="roslaunch pylon_camera ${launch_file}.launch ${param}"
  NUM_PROCESSES="$(pgrep -c -f "${launch_file}")"
  if [ "${NUM_PROCESSES}" -eq 0 ]; then
    eval "ROS_LOG_DIR=${ROS_LOG_DIR} nohup ${CMD} </dev/null >${LOG} 2>&1 &"
  fi
}

function start_fe_with() {
  param=""
  if [[ "$@" = *"head_left"* ]]; then
    param="${param} head_left:=true"
  else
    param="${param} head_left:=false"
  fi

  if [[ "$@" = *"head_right"* ]]; then
    param="${param} head_right:=true"
  else
    param="${param} head_right:=false"
  fi

  if [[ "$@" = *"mid_left"* ]]; then
    param="${param} mid_left:=true"
  else
    param="${param} mid_left:=false"
  fi

  if [[ "$@" = *"mid_right"* ]]; then
    param="${param} mid_right:=true"
  else
    param="${param} mid_right:=false"
  fi

  if [[ "$@" = *"front_left"* ]]; then
    param="${param} front_left:=true"
  else
    param="${param} front_left:=false"
  fi
  if [[ "$@" = *"front_right"* ]]; then
    param="${param} front_right:=true"
  else
    param="${param} front_right:=false"
  fi

  if [[ "$@" = *"tail_left"* ]]; then
    param="${param} tail_left:=true"
  else
    param="${param} tail_left:=false"
  fi
  if [[ "$@" = *"tail_right"* ]]; then
    param="${param} tail_right:=true"
  else
    param="${param} tail_right:=false"
  fi

  launch_file=""
  if [[ "$@" = *"raw"* ]]; then
    launch_file="run_camera"
  else
    launch_file="run_camera_and_compression"
  fi

  CMD="ROS_LOG_DIR=${ROS_LOG_DIR} roslaunch pylon_camera ${launch_file}.launch ${param}"
  NUM_PROCESSES="$(pgrep -c -f "${launch_file}")"
  if [ "${NUM_PROCESSES}" -eq 0 ]; then
    eval "${CMD}"
  fi
}

function start_without() {
  param=""
  if [[ "$@" = *"head_left"* ]]; then
    param="${param} head_left:=false"
  else
    param="${param} head_left:=true"
  fi

  if [[ "$@" = *"head_right"* ]]; then
    param="${param} head_right:=false"
  else
    param="${param} head_right:=true"
  fi

  if [[ "$@" = *"mid_left"* ]]; then
    param="${param} mid_left:=false"
  else
    param="${param} mid_left:=true"
  fi

  if [[ "$@" = *"mid_right"* ]]; then
    param="${param} mid_right:=false"
  else
    param="${param} mid_right:=true"
  fi

  if [[ "$@" = *"front_left"* ]]; then
    param="${param} front_left:=false"
  else
    param="${param} front_left:=true"
  fi
  if [[ "$@" = *"front_right"* ]]; then
    param="${param} front_right:=false"
  else
    param="${param} front_right:=true"
  fi

  if [[ "$@" = *"tail_left"* ]]; then
    param="${param} tail_left:=false"
  else
    param="${param} tail_left:=true"
  fi
  if [[ "$@" = *"tail_right"* ]]; then
    param="${param} tail_right:=false"
  else
    param="${param} tail_right:=true"
  fi

  launch_file=""
  if [[ "$@" = *"raw"* ]]; then
    launch_file="run_camera"
  else
    launch_file="run_camera_and_compression"
  fi

  CMD="roslaunch pylon_camera ${launch_file}.launch ${param}"
  NUM_PROCESSES="$(pgrep -c -f "${launch_file}")"
  if [ "${NUM_PROCESSES}" -eq 0 ]; then
    eval "ROS_LOG_DIR=${ROS_LOG_DIR} nohup ${CMD} </dev/null >${LOG} 2>&1 &"
  fi
}

function start_fe_without() {
  param=""
  if [[ "$@" = *"head_left"* ]]; then
    param="${param} head_left:=false"
  else
    param="${param} head_left:=true"
  fi

  if [[ "$@" = *"head_right"* ]]; then
    param="${param} head_right:=false"
  else
    param="${param} head_right:=true"
  fi

  if [[ "$@" = *"mid_left"* ]]; then
    param="${param} mid_left:=false"
  else
    param="${param} mid_left:=true"
  fi

  if [[ "$@" = *"mid_right"* ]]; then
    param="${param} mid_right:=false"
  else
    param="${param} mid_right:=true"
  fi

  if [[ "$@" = *"front_left"* ]]; then
    param="${param} front_left:=false"
  else
    param="${param} front_left:=true"
  fi
  if [[ "$@" = *"front_right"* ]]; then
    param="${param} front_right:=false"
  else
    param="${param} front_right:=true"
  fi

  if [[ "$@" = *"tail_left"* ]]; then
    param="${param} tail_left:=false"
  else
    param="${param} tail_left:=true"
  fi
  if [[ "$@" = *"tail_right"* ]]; then
    param="${param} tail_right:=false"
  else
    param="${param} tail_right:=true"
  fi

  launch_file=""
  if [[ "$@" = *"raw"* ]]; then
    launch_file="run_camera"
  else
    launch_file="run_camera_and_compression"
  fi

  CMD="ROS_LOG_DIR=${ROS_LOG_DIR} roslaunch pylon_camera ${launch_file}.launch ${param}"
  echo $CMD
  NUM_PROCESSES="$(pgrep -c -f "${launch_file}")"
  if [ "${NUM_PROCESSES}" -eq 0 ]; then
    eval "${CMD}"
  fi
}


function stop() {
  pkill -f run_camera
  pkill -f run_camera_and_compression
}


function print_usage() {
  RED='\033[0;31m'
  BLUE='\033[0;34m'
  BOLD='\033[1m'
  NONE='\033[0m'

  echo -e "\n${RED}Usage${NONE}:
  .${BOLD}/scripts/camera.sh${NONE} [OPTION]"

  echo -e "\n${RED}Options${NONE}:
  ${BLUE}start${NONE}: start camera and compression by default. Use `start raw` to start camera without compression.
  ${BLUE}start_fe${NONE}: start camera and compression by default. Use `start raw` to start camera without compression.
  ${BLUE}start_with${NONE}: start camera with specific camera. \
    Use `start_with head_left front_right` to start head_left and front_right(camera and compression).\
    Use `start_with raw head_left front_right` to start head_left and front_right(camera without compression).
  ${BLUE}start_fe_with${NONE}: start camera with specific camera. \
    Use `start_with head_left front_right` to start head_left and front_right(camera and compression).\
    Use `start_with raw head_left front_right` to start head_left and front_right(camera without compression).
  ${BLUE}start_without${NONE}: start camera without specific camera. \
    Use `start_with head_left front_right` to start all cameras except head_left or front_right(camera and compression).\
    Use `start_with raw head_left front_right` to start all cameras except head_left or front_right(camera without compression).
  ${BLUE}start_fe_without${NONE}: start camera without specific camera. \
    Use `start_with head_left front_right` to start all cameras except head_left or front_right(camera and compression).\
    Use `start_with raw head_left front_right` to start all cameras except head_left or front_right(camera without compression).
  ${BLUE}*${NONE}: start camera and compression
  "
}

# run command_name module_name
function run() {
  case $1 in
    start_with)
      start_with "$@"
      ;;
    start_without)
      start_without "$@"
      ;;
    start_fe_with)
      start_fe_with "$@"
      ;;
    start_fe_without)
      start_fe_without "$@"
      ;;
    start)
      start "$@"
      ;;
    start_fe)
      start_fe "$@"
      ;;
    stop)
      stop
      ;;
    usage)
      print_usage
      ;;
    *)
      start
      ;;
  esac
}

run "$@"
