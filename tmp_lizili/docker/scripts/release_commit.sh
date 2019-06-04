#!/usr/bin/env bash


SHELL_PATH="$( cd "$( dirname "$0"  )" && pwd  )"
source $SHELL_PATH/set_env.sh

set -x
if [ -z "${DOCKER_REPO}" ]; then
    DOCKER_REPO=roadstarauto/roadstar
fi

TAG=local_release
DOCKER_NAME=roadstar_release

docker commit ${DOCKER_NAME} ${DOCKER_REPO}:${TAG}
