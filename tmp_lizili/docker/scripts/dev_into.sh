#!/usr/bin/env bash
SHELL_PATH="$( cd "$( dirname "$0"  )" && pwd  )"
source $SHELL_PATH/set_env.sh

SHELL=zsh
SSH_AGENT=/tmp/.ssh-agent-$USER/agent.sock


if [ "$1" = "bash" ]; then
  SHELL=bash
fi

if [ -z $DOCKER_NAME ];then
    DOCKER_NAME="${USER}_roadstar_dev"
fi

if [ -S $SSH_AUTH_SOCK ]; then
    mkdir -p $(dirname $SSH_AGENT)
    ln -f $SSH_AUTH_SOCK /tmp/.ssh-agent-$USER/agent.sock 2>/dev/null >/dev/null
fi

xhost +local:root 1>/dev/null 2>&1
docker exec \
    -e COLORTERM=$COLORTERM \
    -e DISPLAY=${DOCKER_DISPLAY} \
    -u $USER \
    -it $DOCKER_NAME \
    /bin/$SHELL
xhost -local:root 1>/dev/null 2>&1
