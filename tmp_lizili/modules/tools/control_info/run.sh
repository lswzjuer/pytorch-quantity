#!/usr/bin/env bash




set -x

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export PYTHONPATH="${DIR}/../../../bazel-genfiles:${PYTHONPATH}"
eval "python ${DIR}/control_info.py $@"
