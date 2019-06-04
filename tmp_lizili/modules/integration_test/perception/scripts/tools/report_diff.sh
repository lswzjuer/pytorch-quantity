#! /usr/bin/env bash

SCRIPT_DIR=$(cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
echo $SCRIPT_DIR
CUR_DIR=$(pwd)
echo $CUR_DIR
TOP_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../../.." && pwd )
echo "top_dir = $TOP_DIR"

RELEASE_ROOT_DIR="/roadstar/release"
ROADSTAR_ROOT_DIR="/roadstar"
if [ $TOP_DIR = $RELEASE_ROOT_DIR ];then
  BIN_PREFIX=$RELEASE_ROOT_DIR
else
  BIN_PREFIX="/roadstar/bazel-bin"
fi

DIFF_BIN="$BIN_PREFIX/modules/integration_test/perception/obstacle/report/report_compare"

function main() {
  if [ $TOP_DIR = $RELEASE_ROOT_DIR ];then
    RUN_OPT="run_with_release"
  else
    RUN_OPT="run_with_code"
  fi
  DIFF_XML="$CUR_DIR/$1"
  echo "DIFF_XML = $DIFF_XML run-opt=$RUN_OPT" 
  eval "$DIFF_BIN $DIFF_XML"
}

main $@

