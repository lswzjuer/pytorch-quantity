#!/usr/bin/env bash

DOKCER_HOME="/home/$USER"
if [ "$USER" == "root" ]; then
    DOCKER_HOME="/root"
fi

# Start XVnc/X/Lubuntu
chmod -f 777 /tmp/.X11-unix
# From: https://superuser.com/questions/806637/xauth-not-creating-xauthority-file (squashes complaints about .Xauthority)

touch ~/.Xauthority
xauth generate :0 . trusted
/opt/TurboVNC/bin/vncserver -SecurityTypes None

# X11_PATH=/tmp/.X11-unix/
# FILE=$(ls -tr $X11_PATH | tail -n 1)
# PORT_NUM=$(expr ${FILE:1} + 5900)

# if [ $? -eq 0 ] ; then
# /opt/noVNC/utils/launch.sh --vnc localhost:${PORT_NUM} --cert /opt/noVNC/self.pem --listen 40001;
# fi
