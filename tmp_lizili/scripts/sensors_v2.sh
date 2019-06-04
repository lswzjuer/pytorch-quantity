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


function run_all() {
    bash scripts/lidar.sh $1
    sleep 1s
    # if you need open more cameras, add like this:
    # --basler_start_with=head_left,head_right
    bash scripts/camera_v2.sh $1
    sleep 1s
    bash scripts/radar.sh $1
    sleep 1s
    bash scripts/gnss.sh $1
    sleep 1s
}

function print_usage() {
  RED='\033[0;31m'
  BLUE='\033[0;34m'
  BOLD='\033[1m'
  NONE='\033[0m'

  echo -e "\n${RED}Usage${NONE}:
  .${BOLD}/scripts/sensors_v2.sh${NONE} [OPTION]"

  echo -e "\n${RED}Options${NONE}:
  ${BLUE}start${NONE}: start all sensors
  ${BLUE}stop${NONE}: stop all sensors
  ${BLUE}print_usage${NONE}: print usage 
  ${BLUE}*${NONE}: start all sensors
  "
}

# run command_name module_name
function run() {
    case $1 in
        start)
            run_all 
            ;;
        stop)
            run_all stop 
            ;;
        print_usage)
            print_usage
            ;;
        *)
            run_all
            ;;
    esac
}

run "$1"
