#!/usr/bin/env bash

###############################################################################
# Copyright 2017 The Roadstar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################


addgroup --gid "$DOCKER_GRP_ID" "$DOCKER_GRP"
adduser --disabled-password --gecos '' "$DOCKER_USER" \
    --uid "$DOCKER_USER_ID" --gid "$DOCKER_GRP_ID" 2>/dev/null
usermod -aG sudo "$DOCKER_USER"
echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
sudo rsync -lrKog --chown=${DOCKER_GRP_ID}:${DOCKER_GRP_ID} /etc/skel/. /home/${DOCKER_USER}
chsh -s /usr/bin/zsh $DOCKER_USER

# bazelrc
touch /home/${DOCKER_USER}/.bazelrc
echo "startup --max_idle_secs=0" >> /home/${DOCKER_USER}/.bazelrc
echo "test --action_env=ROS_MASTER_URI" >> /home/${DOCKER_USER}/.bazelrc

# bashrc
echo 'if [ -e "/roadstar/scripts/roadstar_base.sh" ]; then source /roadstar/scripts/roadstar_base.sh; fi' >> "/home/${DOCKER_USER}/.bashrc"
echo "export LD_PRELOAD=/usr/lib/libtcmalloc.so.4" >> /home/${DOCKER_USER}/.bashrc
echo "export ROS_DOMAIN_ID=$ROS_DOMAIN_ID" >> /home/${DOCKER_USER}/.bashrc
echo "ulimit -c unlimited" >> /home/${DOCKER_USER}/.bashrc

# zshrc
echo 'if [ -e "/opt/roadstar-platform/ros/setup.zsh" ]; then source /opt/roadstar-platform/ros/setup.zsh; fi' >> "/home/${DOCKER_USER}/.zshrc"
echo "source /roadstar/scripts/set_env.sh" >> "/home/${DOCKER_USER}/.zshrc"
echo "export LD_PRELOAD=/usr/lib/libdlfaker.so:/usr/lib/libvglfaker.so:/usr/lib/libtcmalloc.so.4" >> /home/${DOCKER_USER}/.zshrc
echo "export ROS_DOMAIN_ID=$ROS_DOMAIN_ID" >> /home/${DOCKER_USER}/.zshrc
echo "ulimit -c unlimited" >> /home/${DOCKER_USER}/.zshrc

echo "$VEHICLE_NAME" | sudo -Hu ${DOCKER_USER} tee /home/${DOCKER_USER}/.vehicle_name

# setup GPS device
if [ -e /dev/ttyUSB0 ]; then
  chmod a+rw /dev/ttyUSB0
fi
if [ -e /dev/ttyUSB1 ]; then
  chmod a+rw /dev/ttyUSB1
fi
if [ -e /dev/ttyUSB2 ]; then
  chmod a+rw /dev/ttyUSB2
fi

if [ -e /dev/video* ]; then
  chmod a+rw /dev/video*
fi
if [ -e /dev/camera* ]; then
  chmod a+rw /dev/camera*
fi
