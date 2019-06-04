#!/bin/zsh

SCRIPT_DIR=$(cd $( dirname "{BASH_SOURCE[0]}" ) && pwd )
echo "SCRIPT_DIR=$SCRIPT_DIR"

flag=$1
dest="roadstar-$flag"
mv /home/$DOCKER_USER/$dest  /home/$DOCKER_USER/roadstar
cd /
rm -rf roadstar
ln -s /home/$DOCKER_USER/roadstar  roadstar
cd /roadstar
./docker/scripts/k8s/prepare_as_root.sh
mkdir -p /tmp/core
chown $DOCKER_USER:$DOCKER_USER /tmp/core
