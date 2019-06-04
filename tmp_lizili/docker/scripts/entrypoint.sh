#!/bin/bash
grep ${DOCKER_USER} /etc/passwd || /roadstar/scripts/docker_adduser.sh

if ! [ `hostname` == "in_release_docker" ]; then
echo -e 'build --remote_http_cache=http://192.168.3.102:8080\ntest --remote_http_cache=http://192.168.3.102:8080' >> /etc/bazel.bazelrc
sudo -Hu ${DOCKER_USER} /usr/local/code-server/code-server -HN /roadstar &
fi

/roadstar/docker/scripts/setup_camera.sh

service ssh start

sudo -Hu ${DOCKER_USER} /roadstar/docker/scripts/start_desktop.sh

sudo -Hu ${DOCKER_USER} "$@"
