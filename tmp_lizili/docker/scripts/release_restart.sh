#!/usr/bin/env bash
SHELL_PATH="$( cd "$( dirname "$0"  )" && pwd  )"
source $SHELL_PATH/set_env.sh

docker restart roadstar_release
