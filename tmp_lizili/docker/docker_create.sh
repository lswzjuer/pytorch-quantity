#!/usr/bin/env bash

SHELL_PATH="$( cd "$( dirname "$0"  )" && pwd  )"

DOCKER_REPO="docker.fabu.ai:5000/roadtensor/roadtensor"
TIME=$(date  +%Y%m%d_%H%M)
TAG="vision-${TIME}"

docker build -t "${DOCKER_REPO}:${TAG}" \
    -f "${SHELL_PATH}/Dockerfile" \
    "${SHELL_PATH}"

sed -i "s/vision-.*\"/${TAG}\"/g" ${SHELL_PATH}/docker_start.sh
echo "using image ${DOCKER_REPO}:${TAG}"
