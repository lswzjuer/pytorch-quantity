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

DURATION=1m
BUFFER_SIZE=2048
DAT=`date +%Y%m%d`
MINUTE=`date +%H%M`

VEHICLE=`cat $HOME/.vehicle_name`

BAG_DIR_CACHE_PATH="/tmp/record_bag_dir.txt"

BAG_DIR=${ROADSTAR_ROOT_DIR}/data/bag/$DAT/$MINUTE

MSGS="/roadstar/drivers/pylon_camera/camera/frame/head_left/jpg
      /roadstar/drivers/pylon_camera/camera/frame/head_right/jpg
      /roadstar/drivers/pylon_camera/camera/frame/front_left/jpg
      /roadstar/drivers/pylon_camera/camera/frame/front_right/jpg
      /roadstar/drivers/pylon_camera/camera/frame/mid_left/jpg
      /roadstar/drivers/pylon_camera/camera/frame/mid_right/jpg
      /roadstar/drivers/pylon_camera/camera/frame/tail_left/jpg
      /roadstar/drivers/pylon_camera/camera/frame/tail_right/jpg
      /roadstar/drivers/leopard_camera/camera/frame/head_left/jpg
      /roadstar/drivers/leopard_camera/camera/frame/head_right/jpg
      /roadstar/drivers/leopard_camera/camera/frame/mid_left/jpg
      /roadstar/drivers/leopard_camera/camera/frame/mid_right/jpg
      /roadstar/drivers/leopard_camera/camera/frame/tail_left/jpg
      /roadstar/drivers/leopard_camera/camera/frame/tail_right/jpg
      /roadstar/drivers/rslidar/Packets/mid_left
      /roadstar/drivers/rslidar/Packets/top_left
      /roadstar/drivers/rslidar/Packets/top_right
      /roadstar/drivers/rslidar/Packets/mid_right
      /roadstar/drivers/lidar/packets/main
      /roadstar/drivers/lidar/packets/head_mid
      /roadstar/drivers/lidar/packets/tail_mid
      /roadstar/drivers/lidar/packets/tail_left
      /roadstar/drivers/lidar/packets/tail_right
      /roadstar/drivers/lidar/packets/top_left
      /roadstar/drivers/lidar/packets/top_right
      /roadstar/drivers/lidar/packets/head_left
      /roadstar/drivers/lidar/packets/head_right
      /roadstar/drivers/delphi_esr
      /roadstar/drivers/conti_radar
      /roadstar/drivers/conti_radar/head_mid
      /roadstar/drivers/esr
      /roadstar/drivers/rsds
      /velodyne_packets
      /velodyne_packets_1
      /velodyne_packets_2
      /velodyne_packets_3
      /velodyne_packets_4
      /roadstar/drivers/Pandar40/Packets
      /roadstar/drivers/Pandar40p/Packets
      /roadstar/canbus/chassis
      /roadstar/canbus/chassis_detail
      /roadstar/drivers/ins
      /roadstar/drivers/gnss/ins
      /roadstar/localization
      /roadstar/drivers/novatel/raw_data
      /roadstar/drivers/novatel/best_pose
      /roadstar/drivers/novatel/ins_stat
      /roadstar/drivers/asensing/raw_data
      /roadstar/drivers/asensing/ins
      /roadstar/drivers/asensing/ins_stat
      /roadstar/drivers/imu/pluto
      /roadstar/perception/fusion_map
      /roadstar/perception/traffic_light
      /roadstar/perception/lane
      /roadstar/perception/lidar_obstacles
      /roadstar/planning/trajectory
      /roadstar/control/control_command
      /roadstar/control/control_status
      /roadstar/control/control_debug
      /roadstar/monitor
      /roadstar/monitor/system_status
      "

function start() {
  stop
  # Record bag to the largest portable-disk.
  if [[ "$@" == *"--portable-disk"* ]] || [ `hostname` == "in_release_docker" ]; then
    LARGEST_DISK="$(df | grep "/media/" | sort -nr -k 4 | head -n 1 | \
        awk '{print substr($0, index($0, $6))}')"
    if [ ! -z "${LARGEST_DISK}" ]; then
      sudo chmod 777 "${LARGEST_DISK}"
      checkdisk $LARGEST_DISK
      REAL_BAG_DIR="${LARGEST_DISK}/$DAT/$MINUTE"
      if [ ! -d "${REAL_BAG_DIR}" ]; then
        mkdir -p "${REAL_BAG_DIR}"
      fi
      BAG_DIR="${ROADSTAR_ROOT_DIR}/data/bag/portable"
      rm -fr "${BAG_DIR}"
      ln -s "${REAL_BAG_DIR}" "${BAG_DIR}"
    else
      echo "Cannot find portable disk."
      echo "Please make sure your container was started AFTER inserting the disk."
      exit 1
    fi
  fi

  # Create and enter into bag dir.
  if [ ! -e "${BAG_DIR}" ]; then
    mkdir -p "${BAG_DIR}"
  fi

  # Cache the bag dir.
  echo ${BAG_DIR} > $BAG_DIR_CACHE_PATH
  echo "Copy driver conf to: ${BAG_DIR}"

  cp /roadstar/release/meta.ini ${BAG_DIR}
  cp -r resources/calibration ${BAG_DIR}/calibration
  cd "${BAG_DIR}" || exit
  echo "Recording bag to: $(pwd)"
  touch README
  echo "time: "\"$DATE-$MINUTE\">> README
  echo "vehicle_name: "\"$VEHICLE\" >> README

  # Write ros_master_uri in README
  ros_master_ip_port=(${ROS_MASTER_URI//// })
  ros_master_ip=(${ros_master_ip_port[1]//:/ })
  echo "ros_master_ip: "\"$ros_master_ip\" >> README
  echo "ros_ip: "\"$ROS_IP\" >> README
  # Write hdmap name and route name in README
  curr_map=(`python ${ROADSTAR_ROOT_DIR}/modules/tools/hdmap_service/call_mapservice.py GetMap --host ${ros_master_ip}`)
  map_name=${curr_map[1]}
  echo "map_name: "\"$map_name\" >> README
  curr_route=(`python ${ROADSTAR_ROOT_DIR}/modules/tools/hdmap_service/call_mapservice.py GetRoute --host ${ros_master_ip}`)
  route_name=${curr_route[1]}
  echo "route_name: "\"$route_name\" >> README
  # echo "Location: "$LOCATION >> README
  # echo "Purpose: "$PURPOSE >> README
  # echo "Duration: "$DURATION >> README
  # echo "Msgs: "$MSGS >> README
  # echo "Remark: "$REMARK >> README
  # echo "Code version: " >> README
  cd $DIR/..
  #git log --pretty=oneline -n 1 >> ${BAG_DIR}/README
  #git submodule foreach git log --pretty=oneline -n 1 >> ${BAG_DIR}/README

  cd "${BAG_DIR}"
  # Start recording.
  LOG="${ROADSTAR_ROOT_DIR}/data/log/${DATE}/record.out.${TIME_IN_SEC}"
  if [[ "$@" == *"--rosbag"* ]]; then
    NUM_PROCESSES="$(pgrep -c -f "rosbag record")"
    echo "now recording ros bag."
    if [ "${NUM_PROCESSES}" -eq 0 ]; then
      nohup rosbag record --split --duration=$DURATION -b $BUFFER_SIZE \
          $MSGS \
          -o $VEHICLE \
          </dev/null >"${LOG}" 2>&1 &
    fi
  else
    echo "now recording message service bag."
    NUM_PROCESSES="$(pgrep -c -f "tools/record_bag")"
    if [ "${NUM_PROCESSES}" -eq 0 ]; then
      nohup /roadstar/release/scripts/message_service.sh record \
          --save-bag-path=$BAG_DIR/$VEHICLE \
          --log_dir=${ROADSTAR_ROOT_DIR}/data/log/${DATE} \
          --log_link=${ROADSTAR_ROOT_DIR}/data/log \
          </dev/null >"${LOG}" 2>&1 &
    fi
  fi
}

function stop() {
  pkill -SIGINT record
  for i in {1..5}; do 
    echo 'waiting record bag to finish';
    sleep 1;
    pgrep record || return;
  done
  echo 'force stopping record bag';
  pkill -SIGKILL record
}

function checkdisk(){
  FILESYSTEM="$(df -T| grep "$1$" | awk '{print $2}')"
  if [ "$FILESYSTEM" != "ext4" ]; then
    echo "file system should be ext4"
    exit 1
  else
    AVAILABLE="$(df | grep "$1$" | awk '{print $4}')"
    AVAILABLE=`expr $AVAILABLE / 1024 / 1024`
    if [ "$AVAILABLE" -lt 200 ]; then
      echo "$1 available: "$AVAILABLE" GB"
      echo "please make sure at least 200GB for record bag"
      exit 1
    fi
    echo "testing write speed of $1..."
    timeout 5s dd if=/dev/zero of=$1/record_bag_disk_test count=10000000
    WRITESPEED=`du $1/record_bag_disk_test | awk '{print $1}'`
    WRITESPEED=`expr $WRITESPEED / 1024`
    rm $1/record_bag_disk_test
    echo "$1 write speed: "$WRITESPEED" MB/s"
    if [ "$WRITESPEED" -lt 50 ]; then
      echo "please make sure at least 50 MB/s for record bag"
      exit 1
    fi
  fi
}

function help() {
  echo "Usage:"
  echo "$0 [start]                     Record ros bag to /data/bag/DAT."
  echo "$0 [start] --ms                Record message service bag to /data/bag/DAT."
  echo "$0 [start] --portable-disk     Record bag to the largest portable disk (default in release docker)."
  echo "$0 stop                        Stop recording."
  echo "$0 help                        Show this help message."
}

case $1 in
  start)
    shift
    start $@
    ;;
  stop)
    shift
    stop $@
    ;;
  help)
    shift
    help $@
    ;;
  *)
    start $@
    ;;
esac
