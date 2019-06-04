#!/usr/bin/env bash


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${DIR}"

source ${DIR}/scripts/roadstar_base.sh

TIME=$(date  +%Y%m%d_%H%M)
if [ -z "${DOCKER_REPO}" ]; then
    DOCKER_REPO=roadstarauto/roadstar
fi

function print_usage() {
  echo 'Usage:
  ./roadstar_docker.sh [OPTION]'
  echo 'Options:
  build : run build only
  buildify: fix style of BUILD files
  check: run build/lint/test, please make sure it passes before checking in new code
  clean: runs Bazel clean
  coverage: generate test coverage report
  doc: generate doxygen document
  push: pushes the images to Docker hub
  gen: release a docker release image
  lint: run code style check
  release: to build release version
  test: run all the unit tests
  version: current commit and date
  print_usage: prints this menu
  '
}

function start_build_docker() {
  docker ps --format "{{.Names}}" | grep $DOCKER_NAME
  if [ $? != 0 ]; then
    bash docker/scripts/dev_start.sh
  fi
}

function gen_docker() {
  IMG="roadstarauto/roadstar:run-env-20170712_1738"
  RELEASE_DIR=${HOME}/.cache/release
  RELEASE_NAME="${DOCKER_REPO}:release-${TIME}"
  DEFAULT_NAME="${DOCKER_REPO}:release-latest"
  docker pull $IMG

  docker ps -a --format "{{.Names}}" | grep 'roadstar_release' 1>/dev/null
  if [ $? == 0 ];then
    docker stop roadstar_release 1>/dev/null
    docker rm -f roadstar_release 1>/dev/null
  fi
  docker run -it \
      -d \
      --name roadstar_release \
      --net host \
      -v "$RELEASE_DIR:/root/mnt" \
      -w /roadstar \
      "$IMG"

  docker exec roadstar_release bash -c "cp -Lr /root/mnt/* ."
  CONTAINER_ID=$(docker ps | grep roadstar_release | awk '{print $1}')
  docker commit "$CONTAINER_ID" "$RELEASE_NAME"
  docker commit "$CONTAINER_ID" "$DEFAULT_NAME"
  docker stop "$CONTAINER_ID"
}

function push() {
  local DEFAULT_NAME="${DOCKER_REPO}:release-latest"
  local RELEASE_NAME="${DOCKER_REPO}:release-${TIME}"
  docker tag "$DEFAULT_NAME" "$RELEASE_NAME"
  docker push "$DEFAULT_NAME"
  docker push "$RELEASE_NAME"
}

if [ $# != 1 ];then
    print_usage
    exit 1
fi

if [ -z $DOCKER_NAME ];then
    DOCKER_NAME="${USER}_roadstar_dev"
fi

start_build_docker

case $1 in
  print_usage)
    print_usage
    ;;
  push)
    push
    ;;
  gen)
    gen_docker
    ;;
  *)
    docker exec -u $USER ${DOCKER_NAME} bash -c "./roadstar.sh $1"
    ;;
esac
