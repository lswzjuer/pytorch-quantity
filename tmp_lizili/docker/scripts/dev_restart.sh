#!/usr/bin/env bash
SHELL_PATH="$( cd "$( dirname "$0"  )" && pwd  )"
source $SHELL_PATH/set_env.sh

if [ -z $DOCKER_NAME ];then
    DOCKER_NAME="${USER}_roadstar_dev"
fi

docker restart $DOCKER_NAME
