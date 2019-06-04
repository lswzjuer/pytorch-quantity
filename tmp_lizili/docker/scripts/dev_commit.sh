#!/usr/bin/env bash

SHELL_PATH="$( cd "$( dirname "$0"  )" && pwd  )"
source $SHELL_PATH/set_env.sh

set -x
if [ -z "${DOCKER_REPO}" ]; then
    DOCKER_REPO="192.168.3.100:5000/roadstar/roadstar"
fi

TAG=local_dev
DOCKER_NAME=roadstar_dev

docker commit ${DOCKER_NAME} ${DOCKER_REPO}:${TAG}
