#!/usr/bin/env bash

ROADSTAR_ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd)"

ROADSTAR_BIN_PREFIX=${ROADSTAR_ROOT_DIR}
if [ -e "${ROADSTAR_ROOT_DIR}/bazel-bin" ]; then
  ROADSTAR_BIN_PREFIX="${ROADSTAR_ROOT_DIR}/bazel-bin"
fi

function start_log_monitor(){
  # if not find the check_monitor binary file
  if [ ! -e "${ROADSTAR_BIN_PREFIX}/modules/tools/monitor" ]; then
    # not find and build for this target
    
    TARGET="//modules/tools/monitor:log_monitor"
    echo "Start build for target: ${TARGET}"
    bazel build ${TARGET}
  fi

  DATE=$(date +%F)

  LOG_DIR="${ROADSTAR_ROOT_DIR}/data/log/${DATE}"
  
  # check whether the folder is exist    
  if [ ! -e ${LOG_DIR} ]; then
    mkdir ${LOG_DIR}
  fi

  # check if log monitor is already running now
  PROCESS_COUNT="$( ps -aux | grep ${ROADSTAR_BIN_PREFIX}/modules/tools/monitor/log_monitor | grep -v grep | wc -l )"
  if [ ! ${PROCESS_COUNT} -eq 0 ]; then
    echo "log monitor is already running now. Skipping..."
    return 1
  fi

  # not running, then run log monitor
  if [ $? -eq 0 ]; then
    echo "Now you can see the log in ${LOG_DIR}"
  else
    echo "Error Happens. Check if find this target."
  fi

  # run the binary file
  eval "nohup ${ROADSTAR_BIN_PREFIX}/modules/tools/monitor/log_monitor \
        --log_dir=${LOG_DIR} \
        --log_link=${ROADSTAR_ROOT_DIR}/data/log \
        --alsologtostderr=false >/dev/null 2>&1 &"
}

function stop_log_monitor(){
  pkill -f "${ROADSTAR_BIN_PREFIX}/modules/tools/monitor/log_monitor"

  if [ $? -eq 0 ]; then
    echo "Successfully stop log monitor"
  else
    echo "log monitor is not running now -- skip this process"
  fi
}

function run(){
  local cmd=$1
  case ${cmd} in
    start)
      start_log_monitor 
      ;;
    restart)
      stop_log_monitor
      start_log_monitor
      ;;
    stop)
      stop_log_monitor
      ;;
    *)
      start_log_monitor
      ;;
  esac
}

run "$@"
