#!/usr/bin/env bash

SHELL_PATH="$( cd "$( dirname "$0"  )" && pwd  )"
source $SHELL_PATH/set_env.sh

if [ $# == 1 ]; then
  DOCKER_IMG_PATH=$1
fi

VERSION=${VERSION_X86_64}
IMG=${DOCKER_REPO}:${VERSION}
echo "saving docker into ${DOCKER_IMG_PATH}"
docker save ${IMG} -o ${DOCKER_IMG_PATH}
