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


PWDIR=$PWD
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source "${DIR}/roadstar_base.sh"

MESSAGE_SERVICE_DIR=${ROADSTAR_BIN_PREFIX}/modules/common/message/tools

function run() {
    case $1 in
        play)
          shift
          cd $PWDIR
          for x in $@
          do
            if [ -f "$x" ]; then
              FILENAMES="$FILENAMES ""`readlink -f $x`"
            else
              FLAGS="$FLAGS ""$x"
            fi
          done
          cd ${ROADSTAR_ROOT_DIR}
          $MESSAGE_SERVICE_DIR/play_bag $FILENAMES $FLAGS
            ;;
        record)
          shift
          $MESSAGE_SERVICE_DIR/record_bag $@
            ;;
        echo)
          shift
          $MESSAGE_SERVICE_DIR/echo_message $@
            ;;
        diagnose)
          shift
          $MESSAGE_SERVICE_DIR/diagnose $@
            ;;
        *)
          echo "
Sample Usage:
  play BAGS -l -r 0.5 -s 20 -u 10
  record --save_bag_path /tmp/bags
  diagnose --module_name monitor
  echo --type SYSTEM_STATUS"
            ;;
    esac
}

cd ${ROADSTAR_ROOT_DIR}
./scripts/roadstar_config.py
run "$@"

