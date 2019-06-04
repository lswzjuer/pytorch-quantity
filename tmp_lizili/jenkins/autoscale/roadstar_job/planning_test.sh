#!/usr/bin/env bash

SCRIPT_DIR=$(cd $( dirname "{BASH_SOURCE[0]}" ) && pwd )

TEST_BASE_DIR="/roadstar/release/modules/integration_test/simulation"
SCRIPTS_BASE_DIR="${TEST_BASE_DIR}/scripts"
INTEGRATION_XML="param/integration_test.xml"

function backup_records(){
  OUTPUT_PATH=$1
  BAG_DIR=$2
  REPORT=$3
  PLANNING_LOG_OUT=$4
  CONTROL_LOG_OUT=$5
  shift 5

  MASTER_COMMIT_ID=`cat /roadstar/release/meta.ini | grep git_commit`
  MASTER_COMMIT_ID=${MASTER_COMMIT_ID#* }
  TIME=`date +%Y-%m-%d-%H-%M`
  RECORDS_PATH="$MASTER_COMMIT_ID"_"$TIME"
  echo "Jenkins INFO: backup records to $OUTPUT_PATH$RECORDS_PATH"

  cd $OUTPUT_PATH
  mkdir $RECORDS_PATH
  cp -r $BAG_DIR $RECORDS_PATH/bags

  echo $REPORT
  if [ -e $REPORT ];then
    cp $REPORT $RECORDS_PATH/report
  fi

  cp /roadstar/release/meta.ini $RECORDS_PATH/

  cd $RECORDS_PATH
  mkdir log
  cp $PLANNING_LOG_OUT log/.
  cp $CONTROL_LOG_OUT log/.
  cp ${PLANNING_LOG_OUT%/*}/planning.INFO log/.
  cp ${CONTROL_LOG_OUT%/*}/control.INFO log/.

}

cd $SCRIPTS_BASE_DIR
source test.sh ${INTEGRATION_XML}

backup_records $OUTPUT_PATH $BAG_DIR $REPORT_PATH$REPORT_NAME $PLANNING_LOG_OUT $CONTROL_LOG_OUT

if [ -e $REPORT_PATH$REPORT_NAME ];then
  echo "Jenkins INFO: report path = $REPORT_PATH$REPORT_NAME"
else
  echo "Jenkins INFO: generating report failed"
  echo "planning INTEGRATION TEST failed"
  touch $OUTPUT_PATH$RECORDS_PATH/generate_report_failed
  exit 1 
fi

echo "===========report result============"
python $SCRIPTS_BASE_DIR/show_report.py $REPORT_PATH$REPORT_NAME
echo "===================================="

IS_VALID=$(python $SCRIPTS_BASE_DIR/check_report_value.py "$REPORT_PATH$REPORT_NAME")
if [ "$IS_VALID" -eq "1" ];then
  echo "evaluation test failed. Exit now."
  touch $OUTPUT_PATH$RECORDS_PATH/evaluation_test_failed
  exit 1 
fi

echo "planning INTEGRATION TEST finished successfully"
touch $OUTPUT_PATH$RECORDS_PATH/integration_test_success
