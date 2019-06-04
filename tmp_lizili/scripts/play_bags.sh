#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "${DIR}/.."

source "$DIR/roadstar_base.sh"

Bags=`find $1 -name '*.bag' | sort`
shift 1
for bag in ${Bags[@]}; do
  echo $PARAM
  echo "play "$bag
  rosbag play $bag $@ \
  /roadstar/monitor/system_status:=/monitor_null \
  /roadstar/planning/trajectory:=/trajectory_null \
  /roadstar/perception/fusion_map:=/fusion_map_null \
  /roadstar/perception/camera_obstacles:=/camera_obstacles_null \
  /roadstar/perception/lidar_obstacles:=/lidar_obstacles_null \
  /roadstar/perception/traffic_light:=/traffic_light_null \

done
