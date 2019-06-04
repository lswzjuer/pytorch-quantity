#! /usr/bin/env bash

SCRIPT_DIR=$(cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
echo "script_dir = $SCRIPT_DIR"
CUR_DIR=$(pwd)
echo "cur_dir = $CUR_DIR"

source $SCRIPT_DIR/../../common/scripts/base.sh

MODULE_BASE_DIR="$BIN_PREFIX/modules/integration_test/perception"
PERCEPTION_TEST="perception_test"

DATE=$(date +%F)

function start() {
  local MODULE=$1
  local CONF_XML=$2
  local GFLAG=$3
  # echo $MODULE
  shift 1
  LOG="${LOG_BASE_DIR}/data/log/${DATE}/${MODULE}.out"
  if [ ! -e "${LOG_BASE_DIR}/data/log/${DATE}" ];then
    mkdir -p "${LOG_BASE_DIR}/data/log/${DATE}"
  fi
  if [ -e "$MODULE_BASE_DIR/$MODULE" ]; then
    # eval "$BASE_DIR/$MODULE $@"
    eval "nohup ${MODULE_BASE_DIR}/${MODULE} $CONF_XML $GFLAG\
        --log_dir=${LOG_BASE_DIR}/data/log/${DATE} \
        --log_link=${LOG_BASE_DIR}/data/log  </dev/null >${LOG} 2>&1 &"
  else
    echo "Jenkins INFO: $MODULE_BASE_DIR/$MODULE is not exists, please make sure you have build the roadstar project successfully."
    exit 1
  fi
}

function stop() {
  local MODULE=$1
  # echo $MODULE
  shift 1
  $(pkill -9 ${MODULE_BASE_DIR}/${MODULE})
  if [ $? -eq 0 ]; then
    echo "Jenkins INFO: Successfully stopped module ${MODULE}."
  else
    echo "Jenkins INFO: $MODULE is not running,--skipping."
  fi
}

function bag_play() {
  RUN_OPT=$1
  CONF_XML=$2
  PERCEPTION_VERSION=$3
  EVAL_MODE=$4
  echo "EVAL_MODE is $EVAL_MODE"
  if [ "$EVAL_MODE" = "eval" ];then
    LIDAR_PERCEPTION_GFLAGS="--is_on_jenkins_env=true"
  else
    LIDAR_PERCEPTION_GFLAGS="--is_on_jenkins_env=false"
  fi
  shift 4
  echo "Jenkins INFO: run-opt = ${RUN_OPT} conf_xml = ${CONF_XML}"
  if [ ! -e "$BAG_PARAMETER_GENERATER" ]; then
    echo "Jenkins INFO: $BAG_PARAMETER_GENERATER is not exists, please make sure you have build the roadstar project successfully."
    exit 1
  fi

  BAG_PATH=$(eval "$BAG_PARAMETER_GENERATER $GET_PARAMETER_CMD_BAG_PATH $CONF_XML")
  BAGS_NAME=$(eval "$BAG_PARAMETER_GENERATER $GET_PARAMETER_CMD_BAGS_NAME $CONF_XML")
  MAP_NAME=$(eval "$BAG_PARAMETER_GENERATER $GET_PARAMETER_CMD_MAP_NAME $CONF_XML")
  MAP_ROUTING=$(eval "$BAG_PARAMETER_GENERATER $GET_PARAMETER_CMD_MAP_ROUTING $CONF_XML")
  echo "Jenkins INFO: set map_name = $MAP_NAME"
  echo "Jenkins INFO: set map_routing = $MAP_ROUTING"
  cd $SCRIPT_DIR
  if [ -z "$BAGS_NAME" ];then
    echo "Error.Play bag fail cause the bag_name is empty.Exit now."
    exit 1
  fi
  local play_cmd="./play_bag.sh $RUN_OPT $PERCEPTION_VERSION start $MAP_NAME $MAP_ROUTING $LIDAR_PERCEPTION_GFLAGS"
  for BAG_NAME in $BAGS_NAME
  do
    if [ -z "$BAG_NAME" ];then
      echo "Error.Play bag fail cause the bag_name is empty.Exit now."
      exit 1
    fi
    play_cmd="$play_cmd $BAG_PATH/$BAG_NAME"
  done
  echo "====start play bag $play_cmd now====="
  eval "$play_cmd"
}
 
 
function signal_play_completed() {
  local module=$1
  local pid_module=$(eval "pidof $module")
  echo "Jenkins INFO: pid = $pid_module"
  eval "kill -42 $pid_module"
}

function stop_when_finished() {
  RUN_OPT=$1
  CONF_XML=$2
  PERCEPTION_VERSION=$3
  echo -n "genarating report."
  for (( ; ; ))
  do
    NUM="$(pgrep -c -f "${MODULE_BASE_DIR}/${PERCEPTION_TEST}")"
    if [ ${NUM} -eq "0" ]; then
      cd $SCRIPT_DIR
      ./play_bag.sh $RUN_OPT $PERCEPTION_VERSION stop 
      break
    else
      sleep 1
      echo -n "."
    fi
  done
}


function run_module_with_access_checked(){
  local RUN_OPT=$1
  local MODULE=$2
  local RELEASE=""
  if [ $RUN_OPT = "run_with_code" ];then
    cd /roadstar
    if ! [ -f "/roadstar/modules/${MODULE}/BUILD" ];then
      RELEASE="/release"
      echo "Jenkins INFO: Access $MODULE under release Environment"
    else
      echo "Jenkins INFO: Access $MODULE under dev environment"
    fi
  else
    cd /roadstar/release
    echo "Jenkins INFO: Access $MODULE under release Environment"
  fi
  shift 2
  echo "Jenkins INFO: $@"
  eval ".${RELEASE}/scripts/$MODULE.sh $@"
}
