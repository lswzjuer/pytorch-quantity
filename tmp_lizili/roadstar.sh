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

#=================================================
#                   Utils
#=================================================

function source_roadstar_base() {
  DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
  cd "${DIR}"

  source "${DIR}/scripts/roadstar_base.sh"
}

function roadstar_check_system_config() {
  # check operating system
  OP_SYSTEM=$(uname -s)
  case $OP_SYSTEM in
    "Linux")
      echo "System check passed. Build continue ..."

      # check system configuration
      DEFAULT_MEM_SIZE="2.0"
      MEM_SIZE=$(free | grep Mem | awk '{printf("%0.2f", $2 / 1024.0 / 1024.0)}')
      if (( $(echo "$MEM_SIZE < $DEFAULT_MEM_SIZE" | bc -l) )); then
         warning "System memory [${MEM_SIZE}G] is lower than minimum required memory size [2.0G]. Roadstar build could fail."
      fi
      ;;
    "Darwin")
      warning "Mac OS is not officially supported in the current version. Build could fail. We recommend using Ubuntu 14.04."
      ;;
    *)
      error "Unsupported system: ${OP_SYSTEM}."
      error "Please use Linux, we recommend Ubuntu 14.04."
      exit 1
      ;;
  esac
}

function check_machine_arch() {
  # the machine type, currently support x86_64, aarch64
  MACHINE_ARCH=$(uname -m)

  # Generate WORKSPACE file based on marchine architecture
  if [ "$MACHINE_ARCH" == 'x86_64' ]; then
    sed "s/MACHINE_ARCH/x86_64/g" WORKSPACE.in > WORKSPACE
  elif [ "$MACHINE_ARCH" == 'aarch64' ]; then
    sed "s/MACHINE_ARCH/aarch64/g" WORKSPACE.in > WORKSPACE
  else
    fail "Unknown machine architecture $MACHINE_ARCH"
    exit 1
  fi

  #setup vtk folder name for different systems.
  VTK_VERSION=$(find /usr/include/ -type d  -name "vtk-*" | cut -d '-' -f 2)
  sed -i "s/VTK_VERSION/${VTK_VERSION}/g" WORKSPACE
}

function check_can_files() {
  CAN_CARD="fake_can"
  USE_ESD_CAN=false
  USE_SOCKET_CAN=false
  USE_KVASER_CAN=false
  if [ -f /usr/include/canlib.h ]; then
    echo "USE KVASER CAN"
    USE_KVASER_CAN=true
    CAN_CARD="kvaser_can"
  elif [ -f ./third_party/can_card_library/esd_can/include/ntcan.h \
      -a -f ./third_party/can_card_library/esd_can/lib/libntcan.so.4 \
      -a -f ./third_party/can_card_library/esd_can/lib/libntcan.so.4.0.1 ]; then
    echo "USE ESD CAN"
    USE_ESD_CAN=true
    CAN_CARD="esd_can"
  else
    echo "USE SOCKET CAN"
    USE_SOCKET_CAN=true
    CAN_CARD="socket_can"
  fi
}

function generate_build_targets() {
  if [ -z $NOT_BUILD_PERCEPTION ] ; then
    BUILD_TARGETS=$(bazel query //... | grep -v "third_party" | grep -v "release")
  else
    info 'Skip building perception module!'
    BUILD_TARGETS=`bazel query //... except //modules/perception/...`
  fi

  if [ $? -ne 0 ]; then
    fail 'Build failed!'
  fi
  if ! $USE_ESD_CAN; then
      BUILD_TARGETS=$(echo $BUILD_TARGETS |tr ' ' '\n' | grep -v "hwmonitor" | grep -v "esd")
  fi
}

#=================================================
#              Build functions
#=================================================

function build() {
  START_TIME=$(get_now)

  # avoid perception_v2 to break
  ./scripts/update_resources.sh 

  info "Start building, please wait ..."
  generate_build_targets
  info "Building on $MACHINE_ARCH..."

  MACHINE_ARCH=$(uname -m)
  JOB_ARG=""
  if [ "$MACHINE_ARCH" == 'aarch64' ]; then
    JOB_ARG="--jobs=3"
  fi
  echo "$BUILD_TARGETS" | xargs bazel build $JOB_ARG $DEFINES \
    -c $@
  if [ $? -eq 0 ]; then
    success 'Build passed!'
  else
    fail 'Build failed!'
  fi
  find bazel-genfiles/* -type d -exec touch "{}/__init__.py" \;
  # Update task info template on compiling.
  # bazel-bin/modules/data/util/update_task_info --commit_id=$(git rev-parse HEAD)
}

function cibuild() {
  START_TIME=$(get_now)

  echo "Start building, please wait ..."
  generate_build_targets
  echo "Building on $MACHINE_ARCH..."
  BUILD_TARGETS="
  //modules/control
  //modules/dreamview
  //modules/hdmap
  //modules/localization
  //modules/msgs
  //modules/perception
  //modules/planning
  "
  bazel build $DEFINES -c dbg $@ $BUILD_TARGETS
  if [ $? -eq 0 ]; then
    success 'Build passed!'
  else
    fail 'Build failed!'
  fi
}

function roadstar_build_dbg() {
  build "dbg" $@
}

function roadstar_build_opt() {
  build "opt" $@
}

function check() {
  local check_start_time=$(get_now)
  roadstar_build_dbg $@ && run_test && run_lint

  START_TIME=$check_start_time
  if [ $? -eq 0 ]; then
    success 'Check passed!'
    return 0
  else
    fail 'Check failed!'
    return 1
  fi
}

function warn_proprietary_sw() {
  echo -e "${RED}The release built contains proprietary software provided by other parties.${NO_COLOR}"
  echo -e "${RED}Make sure you have obtained proper licensing agreement for redistribution${NO_COLOR}"
  echo -e "${RED}if you intend to publish the release package built.${NO_COLOR}"
  echo -e "${RED}Such licensing agreement is solely between you and the other parties,${NO_COLOR}"
  echo -e "${RED}and is not covered by the license terms of the roadstar project${NO_COLOR}"
  echo -e "${RED}(see file license).${NO_COLOR}"
}

function release() {
  bash "${DIR}/scripts/release.sh"
}

function gen_coverage() {
  bazel clean
  generate_build_targets
  echo "$BUILD_TARGETS" | grep -v "cnn_segmentation_test" | xargs bazel test $DEFINES -c dbg --config=coverage $@
  if [ $? -ne 0 ]; then
    fail 'run test failed!'
  fi

  COV_DIR=data/cov
  rm -rf $COV_DIR
  files=$(find bazel-out/local-dbg/bin/modules/ -iname "*.gcda" -o -iname "*.gcno" | grep -v external)
  for f in $files; do
    target="$COV_DIR/objs/modules/${f##*modules}"
    mkdir -p "$(dirname "$target")"
    cp "$f" "$target"
  done

  files=$(find bazel-out/local-opt/bin/modules/ -iname "*.gcda" -o -iname "*.gcno" | grep -v external)
  for f in $files; do
    target="$COV_DIR/objs/modules/${f##*modules}"
    mkdir -p "$(dirname "$target")"
    cp "$f" "$target"
  done

  lcov --capture --directory "$COV_DIR/objs" --output-file "$COV_DIR/conv.info"
  if [ $? -ne 0 ]; then
    fail 'lcov failed!'
  fi
  lcov --remove "$COV_DIR/conv.info" \
      "external/*" \
      "/usr/*" \
      "bazel-out/*" \
      "*third_party/*" \
      "tools/*" \
      -o $COV_DIR/stripped_conv.info
  genhtml $COV_DIR/stripped_conv.info --output-directory $COV_DIR/report
  echo "Generated coverage report in $COV_DIR/report/index.html"
}

function run_test() {
  START_TIME=$(get_now)

  generate_build_targets
  if [ "$USE_GPU" == "1" ]; then
    echo -e "${RED}Need GPU to run the tests.${NO_COLOR}"
    echo "$BUILD_TARGETS" | xargs bazel test $DEFINES --config=unit_test -c dbg --test_verbose_timeout_warnings $@
  else
    echo "$BUILD_TARGETS" | grep -v "cnn_segmentation_test" | xargs bazel test $DEFINES --config=unit_test -c dbg --test_verbose_timeout_warnings $@
  fi
  if [ $? -eq 0 ]; then
    success 'Test passed!'
    return 0
  else
    fail 'Test failed!'
    return 1
  fi
}

function citest() {
  START_TIME=$(get_now)
  BUILD_TARGETS="
  //modules/planning/integration_tests:garage_test
  //modules/planning/integration_tests:sunnyvale_loop_test
  //modules/control/integration_tests:simple_control_test
  //modules/prediction/container/obstacles:obstacle_test
  //modules/dreamview/backend/simulation_world:simulation_world_service_test
  "
  bazel test $DEFINES --config=unit_test -c dbg --test_verbose_timeout_warnings $@ $BUILD_TARGETS
  if [ $? -eq 0 ]; then
    success 'Test passed!'
    return 0
  else
    fail 'Test failed!'
    return 1
  fi
}

function run_cpp_lint() {
  generate_build_targets
  echo "$BUILD_TARGETS" | xargs bazel test --config=cpplint -c dbg
}

function run_bash_lint() {
  FILES=$(find "${ROADSTAR_ROOT_DIR}" -type f -name "*.sh" | grep -v ros)
  echo "${FILES}" | xargs shellcheck
}

function run_lint() {
  START_TIME=$(get_now)

  # Add cpplint rule to BUILD files that do not contain it.
  for file in $(find modules -name BUILD | \
    xargs grep -l -E 'cc_library|cc_test|cc_binary' | xargs grep -L 'cpplint()')
  do
    sed -i '1i\load("//tools:cpplint.bzl", "cpplint")\n' $file
    sed -i -e '$a\\ncpplint()' $file
  done

  run_cpp_lint

  if [ $? -eq 0 ]; then
    success 'Lint passed!'
  else
    fail 'Lint failed!'
  fi
}

function clean() {
  bazel clean --async
  rm -rf $HOME/.cache/catkin_ws
}

function buildify() {
  START_TIME=$(get_now)

  local buildifier_url=https://github.com/bazelbuild/buildtools/releases/download/0.4.5/buildifier
  wget $buildifier_url -O ~/.buildifier
  chmod +x ~/.buildifier
  find . -name '*BUILD' -type f -exec ~/.buildifier -showlog -mode=fix {} +
  if [ $? -eq 0 ]; then
    success 'Buildify worked!'
  else
    fail 'Buildify failed!'
  fi
  rm ~/.buildifier
}

function build_fe() {
  cd modules/dreamview/frontend
  yarn config set ignore-engines true
  yarn config set registry https://registry.npm.taobao.org
  yarn --prefer-offline
  mkdir proto_bundle
  node_modules/protobufjs/bin/pbjs -t json -p ${ROADSTAR_ROOT_DIR} \
    ${ROADSTAR_ROOT_DIR}/modules/msgs/dreamview/proto/response.proto > \
    proto_bundle/response_proto_bundle.json
  node_modules/protobufjs/bin/pbjs -t json -p ${ROADSTAR_ROOT_DIR} \
    ${ROADSTAR_ROOT_DIR}/modules/msgs/dreamview/proto/point_cloud.proto> \
    proto_bundle/point_cloud_proto_bundle.json
  yarn build
}

function gen_doc() {
  rm -rf docs/doxygen
  doxygen roadstar.doxygen
}

function version() {
  commit=$(git log -1 --pretty=%H)
  date=$(git log -1 --pretty=%cd)
  echo "Commit: ${commit}"
  echo "Date: ${date}"
}

function build_ros_package() {
  package_path=$1

  if [ ! -d ${package_path} ];then
    fail "${package_path} not found"
  fi

  catkin_workspace_base=$HOME/.cache/catkin_ws
  if [ ! -d ${catkin_workspace_base} ];then
    mkdir -p ${catkin_workspace_base}
  fi

  CURRENT_PATH=$(pwd)
  ROS_PATH="/opt/roadstar-platform/ros"
  source "${ROS_PATH}/setup.bash"
  pwd
  catkin_make_isolated --install --source ${package_path} \
    -C ${catkin_workspace_base} --quiet \
    --install-space "${ROS_PATH}" -DCMAKE_BUILD_TYPE=Release \
    --cmake-args --no-warn-unused-cli
  ec=$?
  find "${ROS_PATH}" -name "*.pyc" -print0 | xargs -0 rm -rf
  if [[ $ec != 0 ]]; then
    exit 1
  fi
}

function build_gnss() {
  protoc modules/common/proto/error_code.proto --cpp_out=./
  protoc modules/common/proto/header.proto --cpp_out=./
  protoc modules/common/proto/geometry.proto --cpp_out=./

  protoc modules/msgs/localization/proto/localization.proto --cpp_out=./
  protoc modules/msgs/drivers/proto/gps.proto --cpp_out=./

  build_ros_package modules/drivers/gnss/rt

  rm -rf modules/common/proto/*.pb.cc
  rm -rf modules/common/proto/*.pb.h
  rm -rf modules/msgs/localization/proto/*.pb.cc
  rm -rf modules/msgs/localization/proto/*.pb.h
  rm -rf modules/msgs/drivers/proto/*.pb.cc
  rm -rf modules/msgs/drivers/proto/*.pb.h
}

function build_pylon_camera() {
  build_ros_package modules/msgs/drivers/ros_msg/common_msgs # build msgs
  build_ros_package modules/msgs/drivers/ros_msg/camera_msgs # build msgs
  build_ros_package modules/drivers/pylon_camera/camera_driver
}

function build_leopard_camera() {
  build_ros_package modules/msgs/drivers/ros_msg/common_msgs # build msgs
  build_ros_package modules/msgs/drivers/ros_msg/camera_msgs # build msgs
  build_ros_package modules/drivers/leopard_camera/usb_cam
  build_ros_package modules/drivers/leopard_camera/leopard_compression
}

function build_pylon_camera_additional() {
  # include visiual manager extract
  build_ros_package modules/msgs/drivers/ros_msg/common_msgs # build msgs
  build_ros_package modules/msgs/drivers/ros_msg/camera_msgs # build msgs
  build_ros_package modules/drivers/pylon_camera/camera_additional
}

function build_velodyne() {
  build_ros_package modules/msgs/drivers/ros_msg/common_msgs # build msgs
  build_ros_package modules/msgs/drivers/ros_msg/velodyne_msgs # build msgs
  build_ros_package modules/drivers/velodyne 
}

function build_pandar() {
  build_ros_package modules/msgs/drivers/ros_msg/pandar_msgs # build msgs
  build_ros_package modules/msgs/drivers/ros_msg/pointcloud_msgs # build msgs
  build_ros_package modules/drivers/pandar 
}

function build_rslidar() {
  build_ros_package modules/msgs/drivers/ros_msg/pointcloud_msgs # build msgs
  build_ros_package modules/drivers/rslidar 
}

function build_driver_nodelets() {
  # include launch file for all drivers
  build_ros_package modules/drivers/driver_nodelets
}

function config() {
  ${ROADSTAR_ROOT_DIR}/scripts/configurator.sh
}

function print_usage() {
  RED='\033[0;31m'
  BLUE='\033[0;34m'
  BOLD='\033[1m'
  NONE='\033[0m'

  echo -e "\n${RED}Usage${NONE}:
  .${BOLD}/roadstar.sh${NONE} [OPTION]"

  echo -e "\n${RED}Options${NONE}:
  ${BLUE}build${NONE}: run build with debug information only
  ${BLUE}build_opt${NONE}: build optimized binary for the code
  ${BLUE}build_gpu${NONE}: run build only with Caffe GPU mode support
  ${BLUE}build_gnss${NONE}: build gnss driver
  ${BLUE}build_velodyne${NONE}: build velodyne driver
  ${BLUE}build_pylon_camera${NONE}: build pylon_camera driver
  ${BLUE}build_leopard_camera${NONE}: build leopard_camera driver
  ${BLUE}build_pylon_camera_additional${NONE}: build pylon_camera tool box
  ${BLUE}build_driver_nodelets${NONE}: install launch file for all drivers
  ${BLUE}build_opt_gpu${NONE}: build optimized binary with Caffe GPU mode support
  ${BLUE}build_fe${NONE}: compile frontend javascript code, this requires all the node_modules to be installed already
  ${BLUE}build_no_perception${NONE}: run build build skip building perception module, useful when some perception dependencies are not satisified, e.g., CUDA, CUDNN, LIDAR, etc.
  ${BLUE}build_prof${NONE}: build for gprof support.
  ${BLUE}buildify${NONE}: fix style of BUILD files
  ${BLUE}check${NONE}: run build/lint/test, please make sure it passes before checking in new code
  ${BLUE}clean${NONE}: run Bazel clean
  ${BLUE}config${NONE}: run configurator tool
  ${BLUE}coverage${NONE}: generate test coverage report
  ${BLUE}doc${NONE}: generate doxygen document
  ${BLUE}lint${NONE}: run code style check
  ${BLUE}usage${NONE}: print this menu
  ${BLUE}release${NONE}: build release version
  ${BLUE}test${NONE}: run all unit tests
  ${BLUE}version${NONE}: display current commit and date
  "
}

function main() {
  source_roadstar_base
  roadstar_check_system_config
  check_machine_arch
  check_can_files

  DEFINES="--define ARCH=${MACHINE_ARCH} --define CAN_CARD=${CAN_CARD} 
        --cxxopt=-DUSE_ESD_CAN=${USE_ESD_CAN} 
        --cxxopt=-DUSE_SOCKET_CAN=${USE_SOCKET_CAN} 
        --cxxopt=-DUSE_KVASER_CAN=${USE_KVASER_CAN}" 

  local cmd=$1
  shift

  case $cmd in
    check)
      DEFINES="${DEFINES} --cxxopt=-DCPU_ONLY"
      check $@
      ;;
    build)
      DEFINES="${DEFINES} --cxxopt=-DCPU_ONLY"
      roadstar_build_dbg $@
      ;;
    build_prof)
      DEFINES="${DEFINES} --config=cpu_prof --cxxopt=-DCPU_ONLY"
      roadstar_build_dbg $@
      ;;
    build_no_perception)
      DEFINES="${DEFINES} --cxxopt=-DCPU_ONLY"
      NOT_BUILD_PERCEPTION=true
      roadstar_build_dbg $@
      ;;
    cibuild)
      DEFINES="${DEFINES} --cxxopt=-DCPU_ONLY"
      cibuild $@
      ;;
    build_opt)
      DEFINES="${DEFINES} --cxxopt=-DCPU_ONLY"
      roadstar_build_opt $@
      ;;
    build_gpu)
      DEFINES="${DEFINES} --cxxopt=-DUSE_CAFFE_GPU"
      roadstar_build_dbg $@
      ;;
    build_opt_gpu)
      DEFINES="${DEFINES} --cxxopt=-DUSE_CAFFE_GPU"
      roadstar_build_opt $@
      ;;
    build_fe)
      build_fe
      ;;
    buildify)
      buildify
      ;;
    build_driver)
      build_gnss && build_pylon_camera && build_leopard_camera && build_velodyne && build_driver_nodelets && build_pandar && build_rslidar
      ;;
    build_gnss)
      build_gnss
      ;;
    build_pylon_camera)
      build_pylon_camera
      ;;
    build_leopard_camera)
      build_leopard_camera
      ;;
    build_pylon_camera_additional)
      build_pylon_camera_additional
      ;;
    build_velodyne)
      build_velodyne
      ;;
    build_pandar)
      build_pandar
      ;;
    build_driver_nodelets)
      build_driver_nodelets
      ;;
    build_usbcam)
      build_usbcam
      ;;
    build_rslidar)
      build_rslidar
      ;;
    config)
      config
      ;;
    doc)
      gen_doc
      ;;
    lint)
      run_lint
      ;;
    test)
      DEFINES="${DEFINES} --cxxopt=-DCPU_ONLY"
      run_test $@
      ;;
    citest)
      DEFINES="${DEFINES} --cxxopt=-DCPU_ONLY"
      citest $@
      ;;
    test_gpu)
      DEFINES="${DEFINES} --cxxopt=-DUSE_CAFFE_GPU"
      USE_GPU="1"
      run_test $@
      ;;
    release)
      release 1
      ;;
    release_noproprietary)
      release 0
      ;;
    coverage)
      gen_coverage $@
      ;;
    clean)
      clean
      ;;
    version)
      version
      ;;
    usage)
      print_usage
      ;;
    *)
      print_usage
      ;;
  esac
}

main $@
