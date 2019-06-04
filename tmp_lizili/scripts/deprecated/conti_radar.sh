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

source "${DIR}/roadstar_base.sh"

# run function from roadstar_base.sh
# run module_name command_name
function start_with() {
  if [ "$#" == '1' ]; then
    run_customized_path drivers/conti_radar conti_radar_tail_mid "$1"
    run_customized_path drivers/conti_radar conti_radar_head_mid "$1"
    return 0
  fi
  for arg in $@; do
    if [ "$arg" == 'tail_mid' ]; then
      run_customized_path drivers/conti_radar conti_radar_tail_mid "$1"
    fi
    if [ "$arg" == 'tail_left' ]; then
      run_customized_path drivers/conti_radar conti_radar_tail_left "$1"
    fi
    if [ "$arg" == 'tail_right' ]; then
      run_customized_path drivers/conti_radar conti_radar_tail_right "$1"
    fi
    if [ "$arg" == 'head_mid' ]; then
      run_customized_path drivers/conti_radar conti_radar_head_mid "$1"
    fi
    if [ "$arg" == 'head_left' ]; then
      run_customized_path drivers/conti_radar conti_radar_head_left "$1"
    fi
    if [ "$arg" == 'head_right' ]; then
      run_customized_path drivers/conti_radar conti_radar_head_right "$1"
    fi
  done
}

function start_without() {
  if [ "$#" == '0' ]; then
    run_customized_path drivers/conti_radar conti_radar_tail_mid
    run_customized_path drivers/conti_radar conti_radar_head_mid
    return 0
  fi
  for arg in $@; do
    if [ "$arg" == 'tail_mid' ]; then
      run_customized_path drivers/conti_radar conti_radar_tail_mid
    fi
    if [ "$arg" == 'tail_left' ]; then
      run_customized_path drivers/conti_radar conti_radar_tail_left
    fi
    if [ "$arg" == 'tail_right' ]; then
      run_customized_path drivers/conti_radar conti_radar_tail_right
    fi
    if [ "$arg" == 'head_mid' ]; then
      run_customized_path drivers/conti_radar conti_radar_head_mid
    fi
    if [ "$arg" == 'head_left' ]; then
      run_customized_path drivers/conti_radar conti_radar_head_left
    fi
    if [ "$arg" == 'head_right' ]; then
      run_customized_path drivers/conti_radar conti_radar_head_right
    fi
  done
}

function stop() {
  if [ "$#" == '1' ]; then
    run_customized_path drivers/conti_radar conti_radar_tail_mid stop
    run_customized_path drivers/conti_radar conti_radar_tail_left stop
    run_customized_path drivers/conti_radar conti_radar_tail_right stop
    run_customized_path drivers/conti_radar conti_radar_head_mid stop
    run_customized_path drivers/conti_radar conti_radar_head_left stop
    run_customized_path drivers/conti_radar conti_radar_head_right stop
  else
    for arg in $@; do
      if [ "$arg" == 'tail_mid' ]; then
        run_customized_path drivers/conti_radar conti_radar_tail_mid stop
      fi
      if [ "$arg" == 'tail_left' ]; then
        run_customized_path drivers/conti_radar conti_radar_tail_left stop
      fi
      if [ "$arg" == 'tail_right' ]; then
        run_customized_path drivers/conti_radar conti_radar_tail_right stop
      fi
      if [ "$arg" == 'head_mid' ]; then
        run_customized_path drivers/conti_radar conti_radar_head_mid stop
      fi
      if [ "$arg" == 'head_left' ]; then
        run_customized_path drivers/conti_radar conti_radar_head_left stop
      fi
      if [ "$arg" == 'head_right' ]; then
        run_customized_path drivers/conti_radar conti_radar_head_right stop
      fi
    done
  fi
}

function run() {
  case $1 in
    start_with|start_without|start_fe_with|start_fe_without|start|start_fe)
      start_with "$@"
      ;;
    stop)
      stop "$@"
      ;;
    *)
      start_without "$@"
      ;;
  esac
}

run "$@"
