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

ROADSTAR_ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

RED='\033[0;31m'
YELLOW='\e[33m'
NO_COLOR='\033[0m'

DATE=$(date +%F)
TIME_IN_SEC=$(date +%H%M%S)

function info() {
  (>&2 echo -e "[\e[34m\e[1mINFO\e[0m] $*")
}

function error() {
  (>&2 echo -e "[${RED}ERROR${NO_COLOR}] $*")
}

function warning() {
  (>&2 echo -e "${YELLOW}[WARNING] $*${NO_COLOR}")
}

function ok() {
  (>&2 echo -e "[\e[32m\e[1m OK \e[0m] $*")
}

function print_delim() {
  echo '============================'
}

function get_now() {
  echo $(date +%s)
}

function print_time() {
  END_TIME=$(get_now)
  ELAPSED_TIME=$(echo "$END_TIME - $START_TIME" | bc -l)
  MESSAGE="Took ${ELAPSED_TIME} seconds"
  info "${MESSAGE}"
}

function success() {
  print_delim
  ok "$1"
  print_time
  print_delim
}

function fail() {
  print_delim
  error "$1"
  print_time
  print_delim
  exit -1
}

function check_in_docker() {
  if [ -f /.dockerenv ]; then
    ROADSTAR_IN_DOCKER=true
  else
    ROADSTAR_IN_DOCKER=false
  fi
  export ROADSTAR_IN_DOCKER
}

function set_lib_path() {
  if [ -e "/opt/roadstar-platform/ros/setup.bash" ]; then
     source "/opt/roadstar-platform/ros/setup.bash"
  fi
  source $ROADSTAR_ROOT_DIR/scripts/set_env.sh
}

function create_data_dir() {
  local DATA_DIR=""
  if [ "$RELEASE_DOCKER" != "1" ];then
    DATA_DIR="${ROADSTAR_ROOT_DIR}/data"
  else
    DATA_DIR="${HOME}/data"
  fi
  if [ ! -e "${DATA_DIR}/log/${DATE}" ]; then
    mkdir -p "${DATA_DIR}/log/${DATE}"
  fi

  if [ ! -e "${DATA_DIR}/bag" ]; then
    mkdir -p "${DATA_DIR}/bag"
  fi

  if [ ! -e "${DATA_DIR}/core" ]; then
    mkdir -p "${DATA_DIR}/core"
  fi
}

function determine_bin_prefix() {
  ROADSTAR_BIN_PREFIX=$ROADSTAR_ROOT_DIR
  if [ -e "${ROADSTAR_ROOT_DIR}/bazel-bin" ]; then
    ROADSTAR_BIN_PREFIX="${ROADSTAR_ROOT_DIR}/bazel-bin"
  fi
  export ROADSTAR_BIN_PREFIX
}

function find_device() {
  # ${1} = device pattern
  local device_list=$(find /dev -name "${1}")
  if [ -z "${device_list}" ]; then
    warning "Failed to find device with pattern \"${1}\" ..."
  else
    local devices=""
    for device in $(find /dev -name "${1}"); do
      ok "Found device: ${device}."
      devices="${devices} --device ${device}:${device}"
    done
    echo "${devices}"
  fi
}

function setup_device() {
  # setup CAN device
  if [ ! -e /dev/can0 ]; then
    sudo mknod --mode=a+rw /dev/can0 c 52 0
  fi
  if [ ! -e /dev/can1 ]; then
    sudo mknod --mode=a+rw /dev/can1 c 52 1
  fi
  if [ ! -e /dev/can2 ]; then
    sudo mknod --mode=a+rw /dev/can2 c 52 2
  fi

  MACHINE_ARCH=$(uname -m)
  if [ "$MACHINE_ARCH" == 'aarch64' ]; then
    sudo ip link set can0 type can bitrate 500000
    sudo ip link set can0 up
  fi

  # setup nvidia device
  sudo /sbin/modprobe nvidia
  sudo /sbin/modprobe nvidia-uvm
  if [ ! -e /dev/nvidia0 ];then
    sudo mknod -m 666 /dev/nvidia0 c 195 0
  fi
  if [ ! -e /dev/nvidiactl ];then
    sudo mknod -m 666 /dev/nvidiactl c 195 255
  fi
  if [ ! -e /dev/nvidia-uvm ];then
    sudo mknod -m 666 /dev/nvidia-uvm c 243 0
  fi
  if [ ! -e /dev/nvidia-uvm-tools ];then
    sudo mknod -m 666 /dev/nvidia-uvm-tools c 243 1
  fi

  if [ ! -e /dev/nvidia-uvm-tools ];then
    sudo mknod -m 666 /dev/nvidia-uvm-tools c 243 1
  fi
}

function is_stopped_customized_path() {
  MODULE_PATH=$1
  MODULE=$2
  NUM_PROCESSES="$(pgrep -c -f "modules/${MODULE_PATH}/${MODULE}")"
  if [ "${NUM_PROCESSES}" -eq 0 ]; then
    return 1
  else
    return 0
  fi
}

function start_core() {
  NUM_PROCESSES="$(pgrep -c -f "roscore")"
  if [ "${NUM_PROCESSES}" -eq 0 ]; then
    echo "Start roscore..."
    ROSCORELOG="${ROADSTAR_ROOT_DIR}/data/log/${DATE}/roscore.out"
    nohup roscore </dev/null >"${ROSCORELOG}" 2>&1 &
  else
    echo "Roscore is running..."
  fi
}

function load_ros_param() {
  MODULE=$1
  PARAM_FILE="${ROADSTAR_ROOT_DIR}/modules/${MODULE}/conf/${MODULE}.yaml"
  if [ -f "$PARAM_FILE" ]; then
    echo "Load ros param..."
    rosparam load $PARAM_FILE /${MODULE}
  fi
}

function start_customized_path() {
  ${ROADSTAR_ROOT_DIR}/scripts/roadstar_config.py || exit 1
  start_core
	load_ros_param	

  MODULE_PATH=$1
  MODULE=$2
  shift 2

  LOG="${ROADSTAR_ROOT_DIR}/data/log/${DATE}/${MODULE}.out.${TIME_IN_SEC}"
  is_stopped_customized_path "${MODULE_PATH}" "${MODULE}"
  if [ $? -eq 1 ]; then
    eval "nohup ${ROADSTAR_BIN_PREFIX}/modules/${MODULE_PATH}/${MODULE} \
        --flagfile=${ROADSTAR_ROOT_DIR}/config/modules/${MODULE_PATH}/conf/${MODULE}.conf \
        --log_dir=${ROADSTAR_ROOT_DIR}/data/log/${DATE} \
        --log_link=${ROADSTAR_ROOT_DIR}/data/log $@ </dev/null >${LOG} 2>&1 &"
    sleep 0.5
    ln -fs $LOG ${ROADSTAR_ROOT_DIR}/data/log/${MODULE}.out
    is_stopped_customized_path "${MODULE_PATH}" "${MODULE}"
    if [ $? -eq 0 ]; then
      echo "Launched module ${MODULE}."
      return 0
    else
      echo "Could not launch module ${MODULE}. Is it already built?"
      return 1
    fi
  else
    echo "Module ${MODULE} is already running - skipping."
    return 2
  fi
}

function start() {
	MODULE=$1
  shift
	
  start_customized_path $MODULE $MODULE "$@"
}

function start_prof_customized_path() {
  ${ROADSTAR_ROOT_DIR}/scripts/roadstar_config.py || exit 1
  start_core
	load_ros_param	

  MODULE_PATH=$1
  MODULE=$2
  shift 2

  echo "Make sure you have built with 'bash roadstar.sh build_prof'"
  LOG="${ROADSTAR_ROOT_DIR}/data/log/${DATE}/${MODULE}.out.${TIME_IN_SEC}"
  is_stopped_customized_path "${MODULE_PATH}" "${MODULE}"
  if [ $? -eq 1 ]; then
    PROF_FILE="/tmp/$MODULE.prof"
    rm -rf $PROF_FILE
    BINARY=${ROADSTAR_BIN_PREFIX}/modules/${MODULE_PATH}/${MODULE}
    eval "CPUPROFILE=$PROF_FILE $BINARY \
        --flagfile=${ROADSTAR_ROOT_DIR}/config/modules/${MODULE_PATH}/conf/${MODULE}.conf \
        --log_dir=${ROADSTAR_ROOT_DIR}/data/log/${DATE} \
        --log_link=${ROADSTAR_ROOT_DIR}/data/log $@ </dev/null >${LOG} 2>&1 &"
    sleep 0.5
    ln -fs $LOG ${ROADSTAR_ROOT_DIR}/data/log/${MODULE}.out
    is_stopped_customized_path "${MODULE_PATH}" "${MODULE}"
    if [ $? -eq 0 ]; then
      echo -e "Launched module ${MODULE} in prof mode. \nExport profile by command:"
      echo -e "${YELLOW}google-pprof --pdf $BINARY $PROF_FILE > ${MODULE}_prof.pdf${NO_COLOR}"
      return 0
    else
      echo "Could not launch module ${MODULE}. Is it already built?"
      return 1
    fi
  else
    echo "Module ${MODULE} is already running - skipping."
    return 2
  fi
}

function start_prof() {
  MODULE=$1
  shift

  start_prof_customized_path $MODULE $MODULE "$@"
}

function start_fe_customized_path() {
  ${ROADSTAR_ROOT_DIR}/scripts/roadstar_config.py || exit 1
  start_core
	load_ros_param	

  MODULE_PATH=$1
  MODULE=$2
  shift 2

  eval "${ROADSTAR_BIN_PREFIX}/modules/${MODULE_PATH}/${MODULE} \
      --alsologtostderr=1 \
      --colorlogtostderr=true \
      --flagfile=${ROADSTAR_ROOT_DIR}/config/modules/${MODULE_PATH}/conf/${MODULE}.conf \
      --log_dir=${ROADSTAR_ROOT_DIR}/data/log/${DATE} \
      $@"
}

function start_fe() {
  MODULE=$1
  shift

  start_fe_customized_path $MODULE $MODULE "$@"
}

function start_gdb_customized_path() {
  ${ROADSTAR_ROOT_DIR}/scripts/roadstar_config.py || exit 1
  start_core
	load_ros_param	

  MODULE_PATH=$1
  MODULE=$2
  shift 2

  eval "gdb --args ${ROADSTAR_BIN_PREFIX}/modules/${MODULE_PATH}/${MODULE} \
      --flagfile=${ROADSTAR_ROOT_DIR}/config/modules/${MODULE_PATH}/conf/${MODULE}.conf \
      --log_dir=${ROADSTAR_ROOT_DIR}/data/log/${DATE} \
      --log_link=${ROADSTAR_ROOT_DIR}/data/log $@"
}

function start_gdb() {
  MODULE=$1
  shift

  start_gdb_customized_path $MODULE $MODULE "$@"
}

function stop_customized_path() {
  MODULE_PATH=$1
  MODULE=$2

  pkill -f "modules/${MODULE_PATH}/${MODULE}"
  if [ $? -eq 0 ]; then
    echo "Successfully stopped module ${MODULE}."
  else
    echo "Module ${MODULE} is not running - skipping."
  fi
}

function stop() {
  MODULE=$1
  stop_customized_path $MODULE $MODULE
}

function help() {
  echo "Usage:
  ./$0 [COMMAND]"
  echo "COMMAND:
  help: this help message
  start: start the module in background
  restart: restart the module in background
  restart_fe: restart the module without putting in background
  start_fe: start the module without putting in background
  start_gdb: start the module with gdb
  stop: stop the module
  "
}

function run_customized_path() {
  local module_path=$1
  local module=$2
  local cmd=$3
  shift 3
  case $cmd in
    start)
      start_customized_path $module_path $module "$@"
      ;;
    restart)
      stop_customized_path $module_path $module
      start_customized_path $module_path $module "$@"
      ;;
    restart_fe)
      stop_customized_path $module_path $module
      start_fe_customized_path $module_path $module "$@"
      ;;
    start_fe)
      start_fe_customized_path $module_path $module "$@"
      ;;
    start_gdb)
      start_gdb_customized_path $module_path $module "$@"
      ;;
    start_prof)
      start_prof_customized_path $module_path $module "$@"
      ;;
    stop)
      stop_customized_path $module_path $module
      ;;
    help)
      help
      ;;
    *)
      start_customized_path $module_path $module $cmd "$@"
    ;;
  esac
}

# run command_name module_name
function run() {
  local module_path=$1
  local module=$2
  shift 2
  run_customized_path $module_path $module "$@"
}

check_in_docker
create_data_dir
set_lib_path
determine_bin_prefix

# only ros domain id is the same, ros nodes can receive msgs each other
# export ROS_DOMAIN_ID=1000
