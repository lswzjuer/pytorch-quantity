#!/usr/bin/env bash




if [ $# -lt 2 ]
then
    echo Usage: ./run.sh planning.pb.txt localization.pb.txt
    exit
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export PYTHONPATH="${DIR}/../../../bazel-genfiles:${PYTHONPATH}"
eval "python ${DIR}/main.py $1 $2"
