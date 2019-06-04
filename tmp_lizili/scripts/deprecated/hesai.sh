#!/usr/bin/env bash

###############################################################################
# Copyright 2018 The Roadstar Authors. All Rights Reserved.
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

LOG="${ROADSTAR_ROOT_DIR}/data/log/${DATE}/pandar/pandar_$(date +%F-%H%M%S).out"

ROS_LOG_DIR="${ROADSTAR_ROOT_DIR}/data/log/${DATE}/pandar"

mkdir -p ${HOME}/.ros

if [ ! -e "${ROS_LOG_DIR}" ]; then
  mkdir -p "${ROS_LOG_DIR}"
fi

function start() {
    cmd="roslaunch pandar start_pandar.launch"
    num_processes="$(pgrep -c -f "start_pandar")"
    if [ "${num_processes}" -eq 0 ]; then
       eval "ROS_LOG_DIR=${ROS_LOG_DIR} nohup ${cmd} </dev/null >${LOG} 2>&1 &"
    fi
}

function convert() {
    cmd="roslaunch pandar_pointcloud pandar_convert_nodelet.launch"
    num_processes="$(pgrep -c -f "pandar_convert_nodelet")"
    if [ "${num_processes}" -eq 0 ]; then
       eval "nohup ${cmd} </dev/null >${LOG} 2>&1 &"
    fi
}

function start_fe() {
    cmd="roslaunch pandar start_pandar.launch"
    num_processes="$(pgrep -c -f "start_pandar")"
    if [ "${num_processes}" -eq 0 ]; then
       eval "${cmd}"
    fi
}

function stop() {
    pkill -f start_pandar
}


function print_usage() {
  RED='\033[0;31m'
  BLUE='\033[0;34m'
  BOLD='\033[1m'
  NONE='\033[0m'

  echo -e "\n${RED}Usage${NONE}:
  .${BOLD}/scripts/hesai.sh${NONE} [OPTION]"

  echo -e "\n${RED}Options${NONE}:
  ${BLUE}start${NONE}: start pandar40 sensor
  ${BLUE}start_fe${NONE}: start pandar40 sensor without putting in background
  ${BLUE}convert${NONE}: convert pandar40 packets to pointcloud
  ${BLUE}stop${NONE}: stop pandar40 sensor
  ${BLUE}*${NONE}: start pandar40 sensor
  "
}

# run command_name module_name
function run() {
    case $1 in
        start)
            start
            ;;
        convert)
            convert
            ;;
        start_fe)
            start_fe
            ;;
        stop)
            stop
            ;;
        *)
            start
            ;;
    esac
}

run "$1"
