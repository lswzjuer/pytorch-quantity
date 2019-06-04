#!/usr/bin/env bash

SHELL_PATH="$( cd "$( dirname "$0"  )" && pwd  )"
source $SHELL_PATH/set_env.sh

TIME=$(date  +%Y%m%d_%H%M)
if [ -z "${DOCKER_REPO}" ]; then
    DOCKER_REPO="docker.fabu.ai:5000/roadstar/roadstar"
fi

ROADSTAR_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"

ARCH=$(uname -m)
TAG="dev-${ARCH}-${TIME}"

# Build image from ROADSTAR_ROOT, while use the specified Dockerfile.
docker build -t "${DOCKER_REPO}:${TAG}" \
    -f "${ROADSTAR_ROOT}/docker/dev.${ARCH}.16.04.dockerfile" \
    "${ROADSTAR_ROOT}"/docker

#rm bazel-0.5.4-installer-linux-x86_64.sh opencv-2.4.13.tar.gz opencv-3.3.1.zip protobuf-cpp-3.4.0.tar.gz

# replace the started device with the latest built image
sed -i "s/dev-${ARCH}-.*\"/${TAG}\"/g" ${ROADSTAR_ROOT}/docker/scripts/dev_start.sh
