
#!/usr/bin/env bash

SHELL=zsh
if [ "$1" = "bash" ]; then
  SHELL=bash
fi

if [ -z $DOCKER_NAME ];then
    DOCKER_NAME="${USER}_docker"
fi

xhost +local:root 1>/dev/null 2>&1
docker exec \
    -u $USER \
    -it ${DOCKER_NAME} \
    /bin/$SHELL
xhost -local:root 1>/dev/null 2>&1
