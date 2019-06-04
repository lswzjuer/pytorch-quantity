#!/usr/bin/env bash



set -x

TIME=$(date  +%Y%m%d_%H%M)

if [ -z "${DOCKER_REPO}" ]; then
    DOCKER_REPO=roadstarauto/roadstar
fi

ROADSTAR_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
TAG="run-env-${TIME}"

# Build image from ROADSTAR_ROOT, while use the specified Dockerfile.
docker build -t "${DOCKER_REPO}:${TAG}" \
    -f "${ROADSTAR_ROOT}/docker/run_env.dockerfile" \
    "${ROADSTAR_ROOT}"

sed -i "s/run-env.*\"/${TAG}\"/g" ${ROADSTAR_ROOT}/roadstar_docker.sh
