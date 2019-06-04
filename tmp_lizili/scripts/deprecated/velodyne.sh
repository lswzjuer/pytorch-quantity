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


LOG="${ROADSTAR_ROOT_DIR}/data/log/${DATE}/velodyne/velodyne_$(date +%F-%H%M%S).out"

ROS_LOG_DIR="${ROADSTAR_ROOT_DIR}/data/log/${DATE}/velodyne"

mkdir -p ${HOME}/.ros

if [ ! -e "${ROS_LOG_DIR}" ]; then
  mkdir -p "${ROS_LOG_DIR}"
fi

function start() {
    cmd="roslaunch velodyne start_velodyne_all.launch"
    num_processes="$(pgrep -c -f "start_velodyne_all")"

    
    if [ "${num_processes}" -eq 0 ]; then
       eval "ROS_LOG_DIR=${ROS_LOG_DIR} nohup ${cmd} </dev/null >${LOG} 2>&1 &"
    fi
}

function start_all() {
    cmd="roslaunch velodyne multi_velodyne.launch"
    num_processes="$(pgrep -c -f "multi_velodyne")"
    if [ "${num_processes}" -eq 0 ]; then
       eval "nohup ${cmd} </dev/null >${LOG} 2>&1 &"
    fi
}

function convert() {
    cmd="roslaunch velodyne_pointcloud convert_nodelet_all.launch"
    num_processes="$(pgrep -c -f "convert_nodelet_all")"
    if [ "${num_processes}" -eq 0 ]; then
       eval "nohup ${cmd} </dev/null >${LOG} 2>&1 &"
    fi
}

function convert_fe() {
    cmd="roslaunch velodyne_pointcloud convert_nodelet_all.launch"
    num_processes="$(pgrep -c -f "convert_nodelet_all")"
    if [ "${num_processes}" -eq 0 ]; then
       eval "${cmd}"
    fi
}

function start_64() {
    CMD="roslaunch velodyne start_velodyne.launch"
    NUM_PROCESSES="$(pgrep -c -f "start_velodyne")"
    if [ "${NUM_PROCESSES}" -eq 0 ]; then
       eval "nohup ${CMD} </dev/null >${LOG} 2>&1 &"
    fi
}

function convert_64() {
    CMD="roslaunch velodyne convert_nodelet.launch"
    NUM_PROCESSES="$(pgrep -c -f "convert_nodelet")"
    if [ "${NUM_PROCESSES}" -eq 0 ]; then
       eval "nohup ${CMD} </dev/null >${LOG} 2>&1 &"
    fi
}

function start_16() {
    CMD="roslaunch velodyne two_velodyne_16.launch"
    NUM_PROCESSES="$(pgrep -c -f "two_velodyne_16")"
    if [ "${NUM_PROCESSES}" -eq 0 ]; then
       eval "nohup ${CMD} </dev/null >${LOG} 2>&1 &"
    fi
}

function convert_16() {
    CMD="roslaunch velodyne two_convert_nodelet_16.launch"
    NUM_PROCESSES="$(pgrep -c -f "two_convert_nodelet_16")"
    if [ "${NUM_PROCESSES}" -eq 0 ]; then
       eval "nohup ${CMD} </dev/null >${LOG} 2>&1 &"
    fi
}

function start_fe() {
    LOG="${ROADSTAR_ROOT_DIR}/data/log/$DATE/velodyne.out"
    CMD="roslaunch velodyne start_velodyne_all.launch"
    NUM_PROCESSES="$(pgrep -c -f "start_velodyne_all")"
    if [ "${NUM_PROCESSES}" -eq 0 ]; then
       eval "${CMD}"
    fi
}

function start_fe_64() {
    LOG="${ROADSTAR_ROOT_DIR}/data/log/$DATE/velodyne.out"
    CMD="roslaunch velodyne start_velodyne.launch"
    NUM_PROCESSES="$(pgrep -c -f "start_velodyne")"
    if [ "${NUM_PROCESSES}" -eq 0 ]; then
       eval "${CMD}"
    fi
}

function start_fe_16() {
    LOG="${ROADSTAR_ROOT_DIR}/data/log/$DATE/velodyne.out"
    CMD="roslaunch velodyne two_velodyne_16.launch"
    NUM_PROCESSES="$(pgrep -c -f "two_velodyne_16")"
    if [ "${NUM_PROCESSES}" -eq 0 ]; then
       eval "${CMD}"
    fi
}

function stop() {
    pkill -f start_velodyne_all
    pkill -f start_velodyne
    pkill -f two_velodyne_16
    pkill -f multi_velodyne
    pkill -f convert_nodelet
    pkill -f convert_nodelet_all
    pkill -f two_convert_nodelet_16
}

function stop_64() {
    pkill -f start_velodyne
}

function stop_16() {
    pkill -f two_velodyne_16
}

function stop_convert() {
  pkill -f convert_nodelet_all
  pkill -f convert_nodelet
  pkill -f two_convert_nodelet_16
}

function print_usage() {
  RED='\033[0;31m'
  BLUE='\033[0;34m'
  BOLD='\033[1m'
  NONE='\033[0m'

  echo -e "\n${RED}Usage${NONE}:
  .${BOLD}/scripts/velodyne.sh${NONE} [OPTION]"

  echo -e "\n${RED}Options${NONE}:
  ${BLUE}start${NONE}: start velodyne64 and two back velodyne16 sensors
  ${BLUE}start_all${NONE}: start velodyne64 and four velodyne16 sensors
  ${BLUE}start_fe${NONE}: start velodyne64 and two back velodyne16 sensors without putting in background
  ${BLUE}convert${NONE}: convert velodyne64 and velodyne16 packets to pointcloud
  ${BLUE}convert_fe${NONE}: convert without putting in background
  ${BLUE}start_64${NONE}: start velodyne64 sensor
  ${BLUE}start_fe_64${NONE}: start velodyne64 sensor without putting in background
  ${BLUE}convert_64${NONE}: convert velodyne64 packets to pointcloud
  ${BLUE}start_16${NONE}: start two back velodyne16 sensors
  ${BLUE}start_16_fe${NONE}: start two back velodyne16 sensors without putting in background
  ${BLUE}convert_16${NONE}: convert two back velodyne16 packets to pointcloud
  ${BLUE}stop${NONE}: stop velodyne64 and two back velodyne16 sensors
  ${BLUE}stop_64${NONE}: stop velodyne64 sensor
  ${BLUE}stop_16${NONE}: stop two back velodyne16 sensors
  ${BLUE}stop_convert${NONE}: stop convert velodyne64 and two back velodyne16 from packets to pointcloud
  ${BLUE}*${NONE}: start velodyne64 and two back velodyne16 sensors
  "
}

# run command_name module_name
function run() {
    case $1 in
        start)
            start
            ;;
        start_all)
            start_all
            ;;
        convert)
            convert
            ;;
        convert_fe)
            convert_fe
            ;;
        start_64)
            start_64
            ;;
        convert_64)
            convert_64
            ;;
        start_16)
            start_16
            ;;
        convert_16)
            convert_16
            ;;
        start_fe)
            start_fe
            ;;
        start_fe_64)
            start_fe_64
            ;;
        start_fe_16)
            start_fe_16
            ;;
        stop)
            stop
            ;;
        stop_16)
            stop_16
            ;;
        stop_64)
            stop_64
            ;;
        stop_convert)
            stop_convert
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
