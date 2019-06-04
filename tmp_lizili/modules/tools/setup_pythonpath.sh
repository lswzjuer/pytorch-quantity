#!/usr/bin/env bash




DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
GEN_PY_PATH="${DIR}/../../bazel-genfiles"
if [ ! -d "${GEN_PY_PATH}" ]; then
    echo 'Please run `bash roadstar.sh build` in roadstar root directory'
else
    export PYTHONPATH=":${PYTHONPATH}"
fi
