#! /usr/bin/env bash

SCRIPT_DIR=$(cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
source ${SCRIPT_DIR}/base.sh

function help() {
  echo "Jenkins INFO: Usage: 
  ./$0 [PARAMS]"
  echo "Jenkins INFO: only one param is needed which is the config xml path of the bag."
}


function  prepare_work() {
  #replace tl_locations.pb.txt cause the traffic light detection is depend on old one.
  RESOURCES_PATH="/roadstar/resources"
  if [ -e "$RESOURCES_PATH/perception/models/tl_detector/tl_locations.pb.txt" ];then
    echo "tl_locations.pb.txt exists,remove now."
    rm "$RESOURCES_PATH/perception/models/tl_detector/tl_locations.pb.txt"
  fi
  cp $RESOURCES_PATH/integration_test/common/demarcate/tl_locations.pb.txt $RESOURCES_PATH/perception/models/tl_detector/tl_locations.pb.txt
}

function after_test(){
  # reset resource
  RESOURCES_PATH="/roadstar/resources"
  cd $RESOURCES_PATH
  git reset --hard HEAD
}

function show_report_res(){
  local CONF_XML=$1
  TEST_OBJECT=$(eval "$BAG_PARAMETER_GENERATER $GET_PARAMETER_CMD_TEST_OBJECT $CONF_XML")
  IS_TEST_OBSTACLE=$((${TEST_OBJECT} & 1))
  if [ "$IS_TEST_OBSTACLE" -gt "0" ];then
    echo "=======obstacle test result begin========"
    REPORT_PATH=$(eval "$BAG_PARAMETER_GENERATER $GET_PARAMETER_CMD_REPORT_PATH $CONF_XML")
    REPORT_NAME=$(eval "$BAG_PARAMETER_GENERATER $GET_PARAMETER_CMD_REPORT_NAME $CONF_XML")
    python $SCRIPT_DIR/show_report.py $REPORT_PATH/$REPORT_NAME precision_average recall_average velocity_precision_p50 velocity_precision_p95\
    velocity_diff_norm_average  frames label_frames perception_frames
    echo "=======obstacle test result end========"
  fi
  if [ "$((${TEST_OBJECT} & 0x2))" -gt "0" ];then
    echo "=======traffic light test result begin========"
    REPORT_PATH=$(eval "$BAG_PARAMETER_GENERATER $GET_PARAMETER_CMD_TRAFFIC_LIGHT_REPORT_PATH $CONF_XML")
    REPORT_NAME=$(eval "$BAG_PARAMETER_GENERATER $GET_PARAMETER_CMD_TRAFFIC_LIGHT_REPORT_NAME $CONF_XML")
    python $SCRIPT_DIR/show_report.py $REPORT_PATH$REPORT_NAME countdown_time_precision countdown_time_recall labeled_traffic_lights_total match_traffic_lights\
    perception_traffic_lights_total precision_average recall_average\
    total_labeled_frames total_match_frames 
    echo "=======traffic light test result end========"
  fi
}

function main() {
  CONF_XML="$CUR_DIR/$1"
  GFLAG_CONF_XML="--integration_test_config_xml=$CONF_XML"
  echo "GFLAG_CONF_XML=$GFLAG_CONF_XML"
  if [ $TOP_DIR = $RELEASE_ROOT_DIR ];then
    RUN_OPT="run_with_release"
  else
    RUN_OPT="run_with_code"
  fi
  echo "Jenkins INFO: conf-xml = $CONF_XML run-opt=$RUN_OPT" 
  if [ $RUN_OPT = "run_with_code" ]; then
    build build_opt_gpu
    if [ ! "$?" -eq 0 ];then
      echo "integration_test terminated."
      exit 1 
    fi
  fi

  PERCEPTION_VERSION=$(eval "$BAG_PARAMETER_GENERATER $GET_PARAMETER_CMD_PERCEPTION_VERSION $CONF_XML")
  if [ -z "$PERCEPTION_VERSION" ];then
    PERCEPTION_VERSION="1"
  fi
  echo "=============Test begin.Perception v${PERCEPTION_VERSION}==================="
  OLD_VEHICLE=`cat /home/${DOCKER_USER}/.vehicle_name`
  EVAL_MODE=$2
  echo "Jenkins INFO: Jenkins INFO: before modify the ~/.vehicle_name=$OLD_VEHICLE"
  VEHICLE="truck_jenkins";
  echo "$VEHICLE" > /home/${DOCKER_USER}/.vehicle_name
  export FORCE_ON_BOARD="false"
  GFLAG_CONF_FILE="--integration_test_config_file=modules/integration_test/perception/conf/integration_deep.conf"
  if [ "$EVAL_MODE" = "eval" ];then
    GFLAG_CONF_FILE="--integration_test_config_file=modules/integration_test/perception/conf/deep_lidar_eval.conf"
  fi
  echo "Jenkins INFO: after modify the ~/.vehicle_name=$VEHICLE"
  echo "Jenkins INFO: GFLAG_CONF_FILE=$GFLAG_CONF_FILE"
  export DRIVE_SCENE=$(eval "$BAG_PARAMETER_GENERATER $GET_PARAMETER_CMD_DIRVE_SCENE $CONF_XML")
  echo "DRIVE_SCENE=$DRIVE_SCENE"

  prepare_work $RUN_OPT
  enter_env $RUN_OPT

  MID_FILE_PATH=$(eval "$BAG_PARAMETER_GENERATER $GET_PARAMETER_CMD_MID_FILE_PATH $CONF_XML")
  echo "Jenkins INFO: MID_FILE_PATH = $MID_FILE_PATH"
  rm -rf $MID_FILE_PATH
  # cd ${SCRIPT_DIR}
  stop $PERCEPTION_TEST
  # start $PERCEPTION_TEST $CONF_XML $GFLAG_CONF_FILE
  enter_env $RUN_OPT
  GFLAG_CALIBRATION_PATH="--calibration_config_path=resources/calibration/data/shiyan_truck_jenkins"
  ./scripts/integration_test.sh start  $GFLAG_CONF_FILE $GFLAG_CONF_XML $GFLAG_CALIBRATION_PATH 
  sleep 2
  
  if [ "$(pgrep -c -f "${MODULE_BASE_DIR}/${PERCEPTION_TEST}")" -lt "1" ];then
    echo "Jenkins INFO: Start ${MODULE_BASE_DIR}/${PERCEPTION_TEST} failed. Exiting now. Look $LOG for detail."
    after_test
    exit 1
  fi
  bag_play $RUN_OPT $CONF_XML $PERCEPTION_VERSION $EVAL_MODE 
  if [ ! $? -eq 0 ];then
    pkill -9 $PERCEPTION_TEST
    after_test 
    exit 1
  fi
  #make sure hdmap is running
  NUM_PROCESSES="$(pgrep -c -f "modules/hdmap/hdmap")"
  if [ "${NUM_PROCESSES}" -eq "0" ]; then
    run_module_with_access_checked $RUN_OPT hdmap start --xmlmap=$MAP_NAME --routing=$MAP_ROUTING
  fi

  #sleep 2 seconds to make sure recv msgs completed
  sleep 2
  signal_play_completed $PERCEPTION_TEST
  stop_when_finished $RUN_OPT $CONF_XML $PERCEPTION_VERSION

  VEHICLE=$OLD_VEHICLE
  echo "$VEHICLE" > /home/${DOCKER_USER}/.vehicle_name
  echo "Jenkins INFO: finally modify the ~/.vehicle_name=$VEHICLE"
  #show report res
  show_report_res $CONF_XML
  after_test
}

main $@

