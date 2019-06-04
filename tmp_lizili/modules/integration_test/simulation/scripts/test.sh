#! /usr/bin/env bash

function help() {
  echo "Jenkins INFO: Usage:
  $0 [PARAMS]"
  echo "Jenkins INFO: only one param is needed which is the config xml path of the bag"
}

if [ x$1 = x ];then
  help
  exit 1
fi

SCRIPT_DIR=$(cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
CUR_DIR=$(pwd)

cd /roadstar/release
source scripts/record_bag.sh start
cd -

source ${SCRIPT_DIR}/base.sh
get_param "$CUR_DIR/$1"

function main() {
  # modify environment varible
  echo "vehicle = $vehicle"
  OLD_VEHICLE_NAME=`cat /home/${DOCKER_USER}/.vehicle_name`
  echo "Jenkins INFO: before modify the ~/.vehicle_name=$OLD_VEHICLE_NAME"
  VEHICLE_NAME="simulation_powertrain"
  echo "$VEHICLE_NAME" > /home/${DOCKER_USER}/.vehicle_name
  export FORCE_ON_BOARD="false"
  echo "Jenkins INFO: after modify the ~/.vehicle_name=$VEHICLE_NAME"
  
  # module run mode in conf_xml
  echo "Jenkins INFO: conf-xml = $CONF_XML"
  echo " --- run-mode --- "
  echo "server: $SERVER_RUN_MODE"
  echo "powertrain: $POWERTRAIN_RUN_MODE"
  echo "hdmap: $HDMAP_RUN_MODE"
  echo "planning: $PLANNING_RUN_MODE"
  echo "control: $CONTROL_RUN_MODE"
  echo "dreamview: $DREAMVIEW_RUN_MODE"
  echo "node: $NODE_RUN_MODE"
  echo " ----------------"

  if [ $NODE_RUN_MODE == "run_with_release" ];then
    REPORT_PATH=/roadstar/release/$REPORT_PATH
  else
    REPORT_PATH=/roadstar/$REPORT_PATH
  fi

  if [ -e $REPORT_PATH$REPORT_NAME ];then
    rm $REPORT_PATH$REPORT_NAME
  fi
 
  # start simulation world server and related nodes
  echo "Jenkins INFO: map = $MAP route = $ROUTE init_utm_x = $INIT_UTM_X init_utm_y = $INIT_UTM_Y init_yaw = $INIT_YAW"
 
  enter_env $HDMAP_RUN_MODE
  ./scripts/hdmap.sh restart -xmlmap=$MAP -routing=$ROUTE
  sleep 3

  enter_env $SERVER_RUN_MODE
  ./scripts/simulation_world_server.sh restart -auto_traffic_light=true
  sleep 3

  enter_env $POWERTRAIN_RUN_MODE
  ./scripts/powertrain.sh restart
  sleep 3

  enter_env $PLANNING_RUN_MODE
  source scripts/planning.sh restart --v=4
  PLANNING_LOG_OUT=$LOG
  ROADSTAR_ROOT_DIR="/roadstar"
  RELEASE_ROOT_DIR="/roadstar/release"

  enter_env $CONTROL_RUN_MODE
  source scripts/control.sh restart
  CONTROL_LOG_OUT=$LOG
  ROADSTAR_ROOT_DIR="/roadstar"
  RELEASE_ROOT_DIR="/roadstar/release"

  enter_env $DREAMVIEW_RUN_MODE
  ./scripts/dreamview.sh restart

  enter_env $NODE_RUN_MODE
  ./scripts/simulation_world_node.sh restart_fe -simulation_world_grpc_address="localhost:6060" -init_utm_x=$INIT_UTM_X -init_utm_y=$INIT_UTM_Y -init_yaw=$INIT_YAW -evaluation=true -get_traffic_light=true
  
  sleep 3
  stop_when_finished 

  VEHICLE_NAME=$OLD_VEHICLE_NAME
  echo "$VEHICLE_NAME" > /home/${DOCKER_USER}/.vehicle_name
  echo -e "\nJenkins INFO: finally modify the ~/.vehicle_name=$VEHICLE_NAME"

  # stop recording bags
  cd $RELEASE_ROOT_DIR
  ./scripts/record_bag.sh stop
}

main $@
