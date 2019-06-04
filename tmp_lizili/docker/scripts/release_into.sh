#!/usr/bin/env bash

SHELL=zsh
if [ "$1" = "bash" ]; then
  SHELL=bash
fi

if [ ! -z $(echo $DISPLAY | grep 'localhost') ];then
		X11_FORWARDING_SLOT=$(echo $DISPLAY | cut -d. -f1 | cut -d: -f2)
		X11_FORWARDING_COOKIES=$(xauth list | grep "^$(hostname)/unix:${DISPLAY_NUMBER}" | awk '{print $3}')
    docker exec \
        -u $USER \
        -it roadstar_release \
        xauth add in_release_docker/unix:$X11_FORWARDING_SLOT MIT-MAGIC-COOKIE-1 $X11_FORWARDING_COOKIES 1>/dev/null 2>&1
fi


xhost +local:root 1>/dev/null 2>&1
docker exec \
    -e COLORTERM=$COLORTERM \
    -u $USER \
    -it roadstar_release \
    /bin/$SHELL
xhost -local:root 1>/dev/null 2>&1
