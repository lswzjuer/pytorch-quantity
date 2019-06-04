#!/bin/bash
echo /opt/roadstar-platform/ros/lib > /etc/ld.so.conf.d/ros.conf

export ROS_DOMAIN_ID=$(date +%N | cut -c1-7)
export VEHICLE_NAME=kubernetes
cd /roadstar
./scripts/docker_adduser.sh
chown -R ${DOCKER_USER}:${DOCKER_GRP} "/home/${DOCKER_USER}"

echo "MAX_JOBS=$MAX_JOBS"
if [ "$MAX_JOBS" -gt "0" ]; then
  eval "echo 'build --jobs=${MAX_JOBS}' >> /etc/bazel.bazelrc"
  eval "echo 'test --jobs=${MAX_JOBS}' >> /etc/bazel.bazelrc"
fi
echo 'build --remote_http_cache=http://192.168.3.102:8080' >> /etc/bazel.bazelrc
su $DOCKER_USER -c "/roadstar/docker/scripts/k8s/prepare_as_user.sh"

service ssh start
