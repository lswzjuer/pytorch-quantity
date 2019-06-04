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


#DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
#source "${DIR}/roadstar_base.sh"

DURATION=1.5s
BUFFER_SIZE=2048

BAG_DIR_CACHE_PATH="/tmp/record_bag_dir.txt"

VEHICLE=`cat $HOME/.vehicle_name`

#BAG_DIR=${ROADSTAR_ROOT_DIR}/data/bag/$DAT/$MINUTE
BAG_DIR=./calibration_bags/

MSGS=("/roadstar/drivers/pylon_camera/camera/frame/head_left/jpg"
      "/roadstar/drivers/pylon_camera/camera/frame/head_right/jpg"
      "/roadstar/drivers/pylon_camera/camera/frame/front_left/jpg"
      "/roadstar/drivers/pylon_camera/camera/frame/front_right/jpg"
      "/roadstar/drivers/pylon_camera/camera/frame/mid_left/jpg"
      "/roadstar/drivers/pylon_camera/camera/frame/mid_right/jpg"
      "/roadstar/drivers/pylon_camera/camera/frame/tail_left/jpg"
      "/roadstar/drivers/pylon_camera/camera/frame/tail_right/jpg"
      "/roadstar/drivers/leopard_camera/camera/frame/head_left/jpg"
      "/roadstar/drivers/leopard_camera/camera/frame/head_right/jpg"
      "/roadstar/drivers/leopard_camera/camera/frame/mid_left/jpg"
      "/roadstar/drivers/leopard_camera/camera/frame/mid_right/jpg"
      "/roadstar/drivers/leopard_camera/camera/frame/tail_left/jpg"
      "/roadstar/drivers/leopard_camera/camera/frame/tail_right/jpg"
      "/roadstar/drivers/rslidar/PointCloud/mid_left"
      "/roadstar/drivers/rslidar/PointCloud/mid_right"
      "/roadstar/drivers/rslidar/PointCloud/top_left"
      "/roadstar/drivers/rslidar/PointCloud/top_right"
      "/roadstar/drivers/lidar/pointcloud/main"
      "/roadstar/drivers/lidar/pointcloud/head_mid"
      "/roadstar/drivers/lidar/pointcloud/tail_mid"
      "/roadstar/drivers/lidar/pointcloud/tail_left"
      "/roadstar/drivers/lidar/pointcloud/tail_right"
      "/roadstar/drivers/lidar/pointcloud/head_left"
      "/roadstar/drivers/lidar/pointcloud/head_right"
      "/roadstar/drivers/lidar/pointcloud/top_left"
      "/roadstar/drivers/lidar/pointcloud/top_right"
      "/roadstar/drivers/delphi_esr"
      "/roadstar/drivers/conti_radar"
      "/roadstar/drivers/conti_radar/head_mid"
      "/roadstar/drivers/velodyne64/PointCloud"
      "/roadstar/drivers/velodyne16/PointCloud_1"
      "/roadstar/drivers/velodyne16/PointCloud_2"
      "/roadstar/drivers/velodyne16/PointCloud_3"
      "/roadstar/drivers/velodyne16/PointCloud_4"
      "/roadstar/drivers/Pandar40/PointCloud"
      "/roadstar/drivers/ins"
      "/roadstar/localization"
      "/roadstar/drivers/novatel/raw_data"
      "/roadstar/drivers/novatel/best_pose"
      "/roadstar/drivers/novatel/ins_stat"
      "/roadstar/drivers/asensing/ins"
      "/roadstar/drivers/asensing/ins_stat")
      
# Create and enter into bag dir.
if [ ! -e "${BAG_DIR}" ]; then
  mkdir -p "${BAG_DIR}"
fi

# Cache the bag dir.
echo ${BAG_DIR} > $BAG_DIR_CACHE_PATH

cd "${BAG_DIR}" || exit
echo "Recording bag to: $(pwd)"

# Start recording.
LOG="/tmp/roadstar_record.out"
NUM_PROCESSES="$(pgrep -c -f "rosbag record")"
if [ "${NUM_PROCESSES}" -eq 0 ]; then
  nohup rosbag record --duration=$DURATION -b ${BUFFER_SIZE} \
      "${MSGS[@]}" \
      -o "${VEHICLE}" \
      </dev/null >"${LOG}" 2>&1 &
fi

sleep "${DURATION}"
