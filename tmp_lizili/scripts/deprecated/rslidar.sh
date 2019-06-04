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

LOG="${ROADSTAR_ROOT_DIR}/data/log/${DATE}/rslidar/rslidar_$(date +%F-%H%M%S).out"

ROS_LOG_DIR="${ROADSTAR_ROOT_DIR}/data/log/${DATE}/rslidar"

mkdir -p ${HOME}/.ros

if [ ! -e "${ROS_LOG_DIR}" ]; then
  mkdir -p "${ROS_LOG_DIR}"
fi

function start() {
    cmd="roslaunch rslidar start_rslidar_all.launch"
    num_processes="$(pgrep -c -f "start_rslidar_all")"
    if [ "${num_processes}" -eq 0 ]; then
       eval "ROS_LOG_DIR=${ROS_LOG_DIR} nohup ${cmd} </dev/null >${LOG} 2>&1 &"
    fi
}

function start_fe() {
    cmd="roslaunch rslidar start_rslidar_all.launch"
    num_processes="$(pgrep -c -f "start_rslidar_all")"
    if [ "${num_processes}" -eq 0 ]; then
       eval "${cmd}"
    fi
}

function stop() {
    pkill -f start_rslidar_all
    pkill -f rslidar_convert_nodelet_all
}


function print_usage() {
  RED='\033[0;31m'
  BLUE='\033[0;34m'
  BOLD='\033[1m'
  NONE='\033[0m'

  echo -e "\n${RED}Usage${NONE}:
  .${BOLD}/scripts/rslidar.sh${NONE} [OPTION]"

  echo -e "\n${RED}Options${NONE}:
  ${BLUE}start${NONE}: start rslidar sensor
  ${BLUE}start_fe${NONE}: start rslidar sensor without putting in background
  ${BLUE}stop${NONE}: stop rslidar sensor
  ${BLUE}*${NONE}: start rslidar sensor
  "
}

# run command_name module_name
function run() {
    case $1 in
        start)
            start
            ;;
        start_fe)
            start_fe
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

run "$1"
