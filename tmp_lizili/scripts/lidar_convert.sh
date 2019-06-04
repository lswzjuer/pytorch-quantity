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

VEHICLE=`cat $HOME/.vehicle_name`

source "${DIR}/roadstar_base.sh"


function run_all() {
    bash scripts/lidar.sh $1 --onboard=false
}

function print_usage() {
  RED='\033[0;31m'
  BLUE='\033[0;34m'
  BOLD='\033[1m'
  NONE='\033[0m'

  echo -e "\n${RED}Vehicle Name: $VEHICLE ${NONE}"
  echo "Please add a vehicle name"

  echo -e "\n${RED}Usage${NONE}:
  .${BOLD}/scripts/lidar_convert.sh${NONE} [Options]"

  echo -e "\n${RED}Example${NONE}:
  .${BOLD}/scripts/lidar_convert.sh truck1${NONE}
  .${BOLD}/scripts/lidar_convert.sh stop${NONE}
  "

  echo -e "\n${RED}Vehicle names${NONE}:
  ${BLUE}truck*${NONE} 
  ${BLUE}hongqi*${NONE}
  ${BLUE}trumpchi*${NONE} 
  "

  echo -e "\n${RED}Options${NONE}:
  ${BLUE}vehicle name${NONE}: start lidar convert
  ${BLUE}stop${NONE}: stop lidar convert
  ${BLUE}*${NONE}: print usage 
  "
}

# run command_name module_name
function run() {
    case $1 in
        truck*|hongqi*|trumpchi*)
            echo "$1" > $HOME/.vehicle_name && run_all restart
            ;;
        stop)
            run_all stop 
            ;;
        *)
            print_usage
            ;;
    esac
}

run "$@"
