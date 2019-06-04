#!/usr/bin/env bash

set -x

# remove the old versions docker.
sudo apt-get remove docker docker-engine docker.io

sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common

# install docker
curl -fsSL https://mirrors.ustc.edu.cn/docker-ce/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository \
   "deb [arch=amd64] https://mirrors.ustc.edu.cn/docker-ce/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

sudo apt-get update
sudo apt-get -y install docker-ce
sudo usermod -aG docker "$USER"

# install nvidia-docker if the nvidia driver has been installed.
if [ ! -z "$(command -v nvidia-smi)" ]; then
  # If you have nvidia-docker 1.0 installed: we need to remove it and all existing GPU containers
  docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
  sudo apt-get purge -y nvidia-docker

  # Add the package repositories
  curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
    sudo apt-key add -
  distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
  curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
  sudo apt-get update

  # Install nvidia-docker2 and reload the Docker daemon configuration
  sudo apt-get install -y nvidia-docker2
  sudo pkill -SIGHUP dockerd
fi

SHELL_PATH="$( cd "$( dirname "$0"  )" && pwd  )"
if [ -d /etc/docker ]; then
    sudo mkdir -p /etc/docker
fi
sudo cp $SHELL_PATH/daemon.json /etc/docker/
sudo service docker restart
echo "copy daemon.json"
newgrp docker
