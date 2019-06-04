#! /usr/bin/env bash

TOP_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../../.." && pwd)
echo "Jenkins INFO: top_dir = $TOP_DIR"

RELEASE_ROOT_DIR="/roadstar/release"
ROADSTAR_ROOT_DIR="/roadstar"
if [ $TOP_DIR = $RELEASE_ROOT_DIR ];then
  BIN_PREFIX=$RELEASE_ROOT_DIR
  LOG_BASE_DIR=$RELEASE_ROOT_DIR
else
  BIN_PREFIX="/roadstar/bazel-bin"
  LOG_BASE_DIR=$ROADSTAR_ROOT_DIR
fi
echo "Jenkins INFO: bin_prefix=$BIN_PREFIX"
echo "Jenkins INFO: log_base_dir=$LOG_BASE_DIR"

BAG_PARAMETER_GENERATER="$BIN_PREFIX/modules/integration_test/common/xml_param/get_bag_parameter"
GET_PARAMETER_CMD_BAG_PATH="bag_path"
GET_PARAMETER_CMD_VEHICLE="vehicle"
GET_PARAMETER_CMD_BAGS_NAME="bags_name"
GET_PARAMETER_CMD_MAP_NAME="map_name"
GET_PARAMETER_CMD_MAP_ROUTING="routing"
GET_PARAMETER_CMD_CALIBRATION_SOURCE="calibration_source";
GET_PARAMETER_CMD_CALIBRATION_DEST="calibration_dest";
GET_PARAMETER_CMD_REPORT_PATH="report_path";
GET_PARAMETER_CMD_REPORT_NAME="report_name";
GET_PARAMETER_CMD_TRAFFIC_LIGHT_REPORT_PATH="traffic_light_report_path";
GET_PARAMETER_CMD_TRAFFIC_LIGHT_REPORT_NAME="traffic_light_report_name";
GET_PARAMETER_CMD_MID_FILE_PATH="mid_file_path";
GET_PARAMETER_CMD_PERCEPTION_VERSION="perception";
GET_PARAMETER_CMD_DIRVE_SCENE="drive_scene";
GET_PARAMETER_CMD_TEST_OBJECT="test_object";
GET_PARAMETER_CMD_OUTPUT_PATH="output_path"
GET_PARAMETER_CMD_SERVER_RUN_MODE="server_run_mode"
GET_PARAMETER_CMD_POWERTRAIN_RUN_MODE="powertrain_run_mode"
GET_PARAMETER_CMD_HDMAP_RUN_MODE="hdmap_run_mode"
GET_PARAMETER_CMD_PLANNING_RUN_MODE="planning_run_mode"
GET_PARAMETER_CMD_CONTROL_RUN_MODE="control_run_mode"
GET_PARAMETER_CMD_DREAMVIEW_RUN_MODE="dreamview_run_mode"
GET_PARAMETER_CMD_NODE_RUN_MODE="node_run_mode"
GET_PARAMETER_CMD_INIT_UTM_X="init_utm_x"
GET_PARAMETER_CMD_INIT_UTM_Y="init_utm_y"
GET_PARAMETER_CMD_INIT_YAW="init_yaw"

function build() {
  cd $ROADSTAR_ROOT_DIR
  ./roadstar.sh $@
}

function enter_env() {
  RUN_OPT=$1
  if [ $RUN_OPT = "run_with_code" ];then
    cd $ROADSTAR_ROOT_DIR
  else
    cd ${ROADSTAR_ROOT_DIR}/release
  fi
}
  
