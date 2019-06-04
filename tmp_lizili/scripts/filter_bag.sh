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

function print_help() {
   echo "Example: filter_fusionmap.sh -[nf] [-d|--dir] bag1 bag2 bag3 ..."
   echo "-d|--dir the target storage directory"
   echo "This script works in three modes:"
   echo "  -nf filter out fusionmap *.nf.bag"
   echo "  -nc filter out camera *.nc.bag"
}

fusionmap_topic="topic != '/roadstar/perception/fusion_map'"

camera_topic="topic != '/roadstar/drivers/pylon_camera/camera/frame/front_left/jpg' \
          and topic != '/roadstar/drivers/pylon_camera/camera/frame/front_right/jpg' \
          and topic != '/roadstar/drivers/pylon_camera/camera/frame/head_left/jpg' \
          and topic != '/roadstar/drivers/pylon_camera/camera/frame/head_right/jpg' \
          and topic != '/roadstar/drivers/pylon_camera/camera/frame/mid_left/jpg' \
          and topic != '/roadstar/drivers/pylon_camera/camera/frame/mid_right/jpg' \
          and topic != '/roadstar/drivers/pylon_camera/camera/frame/tail_left/jpg' \
          and topic != '/roadstar/drivers/pylon_camera/camera/frame/tail_right/jpg'"

is_no_fusionmap_topic=false
is_no_camera_topic=false

echo $@

#argument parsing code from https://stackoverflow.com/a/14203146
POSITIONAL=()
target_dir=""
while [[ $# -gt 0 ]]; do
key="$1"
case $key in
    -nf|--nofusionmap)
    is_no_fusionmap_topic=true
    work_mode_num=$((work_mode_num+1))
    shift # past argument
    ;;
    -nc|--nocamera)
    is_no_camera_topic=true
    work_mode_num=$((work_mode_num+1))
    shift # past argument
    ;;
    -d|--dir)
    target_dir="$2"
    shift # past argument
    shift # past value
    ;;
    -h|--help)
    print_help
    exit 0
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done

if [[ $work_mode_num -eq 0 ]]; then
   print_help
   exit 0
fi

set -- "${POSITIONAL[@]}" # restore positional parameters


function filter() {
    target=""
    name=$(basename $1)
    if $is_no_fusionmap_topic; then
        target="$2/${name%.*}.nf.bag"
        rosbag filter $1 "$target" "$fusionmap_topic"
    fi
    if $is_no_camera_topic; then
        target="$2/${name%.*}.nc.bag"
        rosbag filter $1 "$target" "$camera_topic"
    fi
    echo "filtered ${bag} to $target"
}

echo $@
for bag in $@; do
   folder=""
   if [ -z $target_dir ] ; then
     folder="$(dirname $bag)"
   else
      folder=$target_dir
   fi
   filter $bag $folder
done
