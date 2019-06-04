#! /usr/bin/env bash

ROADSTAR_ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../../.." && pwd )"
source $ROADSTAR_ROOT_DIR/scripts/roadstar_base.sh
source $ROADSTAR_ROOT_DIR/modules/integration_test/perception/scripts/base.sh

function set_env(){
  RUN_OPT=$1
  case $RUN_OPT in
    run_with_code)
      cd /roadstar/
      echo "Jenkins INFO: cd /roadstar/ "
      ;;
    run_with_release)
      cd /roadstar/release
      echo "Jenkins INFO: cd /roadstar/release"
      ;;
    *)
      cd /roadstar/
      echo "Jenkins INFO: cd /roadstar/ "
      ;;
  esac

}


function start_play() {
  RUN_OPT=$1
  PERCEPTION_VERSION=$2
  MAP_NAME=$3
  MAP_ROUTING=$4
  LIDAR_PERCEPTION_GFLAGS=$5
  shift 5

  run_module_with_access_checked $RUN_OPT hdmap start --xmlmap=$MAP_NAME --routing=$MAP_ROUTING
  sleep 2
  check_process_running hdmap hdmap
  if [ -e "/roadstar/release/scripts/deprecated/velodyne.sh" ];then 
    bash /roadstar/release/scripts/deprecated/velodyne.sh convert
  else
    bash /roadstar/scripts/deprecated/velodyne.sh convert
  fi

  if [ "$PERCEPTION_VERSION" = "2" ];then
    echo "Jenkins INFO:run with perception_v2."
    run_module_with_access_checked $RUN_OPT perception_v2    
    # cd /roadstar/release
    # ./scripts/perception_v2.sh
    sleep 5
    is_stopped_customized_path perception_v2 perception
    if [ ! $? -eq 0 ];then
      echo "start perception failed.exiting now."
      exit 1 
    fi
  else
    echo "Jenkins INFO:run with perception_v1."
    run_module_with_access_checked $RUN_OPT perception    
    sleep 5
    check_process_running perception perception
    echo "Lidar perception gflags $LIDAR_PERCEPTION_GFLAGS"
    run_module_with_access_checked $RUN_OPT lidar_perception $LIDAR_PERCEPTION_GFLAGS
    check_process_running perception/lidar lidar_perception
  fi

  #play bag with quiet mode.
  rosbag play -q $@  /roadstar/perception/fusion_map:=/roadstar/perception/fusion_map_null \
    /roadstar/perception/traffic_light:=/roadstar/null 
}

function check_process_running() {
  MODULE_PATH=$1
  MODULE=$2
  is_stopped_customized_path "${MODULE_PATH}" "${MODULE}"
  if [ $? -eq 0 ]; then
    echo "Jenkins INFO: launched module ${MODULE} sucessfully."
  else
    echo "Jenkins INFO: launched module ${MODULE} failed."
  fi
}


function stop_play() {
  RUN_OPT=$1
  PERCEPTION_VERSION=$2

  run_module_with_access_checked $RUN_OPT hdmap stop
  if [ "$PERCEPTION_VERSION" = "2" ];then
    echo "Jenkins INFO:stop perception_v2."
    run_module_with_access_checked $RUN_OPT perception_v2  stop  
  else
    echo "Jenkins INFO:stop perception_v1."
    run_module_with_access_checked $RUN_OPT perception stop 
    run_module_with_access_checked $RUN_OPT lidar_perception stop 
  fi
  if [ -e "/roadstar/release/scripts/deprecated/velodyne.sh" ];then 
    bash /roadstar/release/scripts/deprecated/velodyne.sh stop
  else
    bash /roadstar/scripts/deprecated/velodyne.sh stop
  fi
}

function main() {
  local RUN_OPT=$1
  local PERCEPTION_VERSION=$2
  local CMD=$3
  shift 3

  case $CMD in
    start)
     stop_play $RUN_OPT $PERCEPTION_VERSION $@
     sleep 5
     start_play $RUN_OPT $PERCEPTION_VERSION $@
     ;;
    stop)
      stop_play $RUN_OPT $PERCEPTION_VERSION $@
      ;;
    *)
      stop_play $RUN_OPT $PERCEPTION_VERSION $@
      ;;
  esac
}

main $@
