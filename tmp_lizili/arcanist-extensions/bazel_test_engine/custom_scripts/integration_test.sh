#! /usr/bin/env bash

EXAMPLE_XML_ONE="param/bags_9000_00-03.xml"
TEST_BASE_DIR="/roadstar/modules/integration_test/perception"
BIN_PREFIX="/roadstar/bazel-bin"
SCRIPTS_BASE_DIR="${TEST_BASE_DIR}/scripts"
BIN_GET_PARAM="$BIN_PREFIX/modules/integration_test/perception/get_bag_parameter"
CMD_GET_REPORT_NAME="report_name"
CMD_GET_REPORT_PATH="report_path"

cd $SCRIPTS_BASE_DIR
./test.sh $EXAMPLE_XML_ONE

REPORT_PATH=$(eval "$BIN_GET_PARAM $CMD_GET_REPORT_PATH ${SCRIPTS_BASE_DIR}/$EXAMPLE_XML_ONE")
REPORT_NAME=$(eval "$BIN_GET_PARAM $CMD_GET_REPORT_NAME ${SCRIPTS_BASE_DIR}/$EXAMPLE_XML_ONE")

IS_VALID=$(python $SCRIPTS_BASE_DIR/check_report_value.py "$REPORT_PATH/$REPORT_NAME")
if [ "$IS_VALID" -eq "1" ];then
  exit 1
else
  exit 0
fi
