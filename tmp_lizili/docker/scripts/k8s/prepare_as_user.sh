#!/bin/zsh
echo "git config --global oh-my-zsh.hide-status 1" >> ~/.zshrc
source ~/.zshrc
cd /roadstar

source docker/scripts/set_env.sh

mkdir ~/.cache

# install ros
mkdir -p $ROS_STORE_PATH
cd $ROS_STORE_PATH/..
rm -rf $ROS_STORE_PATH
wget ${ROS_TAR_URL} 2>/dev/null
tar zxf ros_x86_64.tar.gz
rm ros_x86_64.tar.gz
sudo mkdir -p /opt/roadstar-platform
sudo ln -sf ~/.cache/ros /opt/roadstar-platform/ros
sudo ldconfig

SKEL=/protected/$USER
cp $SKEL/.arcrc ~
cp $SKEL/.gitconfig ~
ln -s $SKEL/.ssh ~
sudo chown -R $USER ~/.ssh/
