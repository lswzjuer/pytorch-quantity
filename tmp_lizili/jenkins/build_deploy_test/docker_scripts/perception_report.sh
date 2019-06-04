#!/usr/bin/env bash

LOG_BACK_UP_PATH="/private/zhangzijian/jenkins_release_log"

ROADSTAR_ROOT_DIR="/roadstar"
cd $ROADSTAR_ROOT_DIR
# backup release log before update_release
function backup_log() {
  if [ -d "$ROADSTAR_ROOT_DIR/release/data/log" ];then
    cp -r "$ROADSTAR_ROOT_DIR/release/data/log/." "$LOG_BACK_UP_PATH"
    echo "Backup release log successfully."
  else
    echo "Backup log fail.No such path."
  fi
}

function upload_integration_result() {
  local json=$1
  echo "upload integration result now."
  curl -d $json -H "Content-Type: application/json" -X POST 192.168.3.113:5000/api/integration
}

function copy_file() {
  local file_source=$1
  local file_dest=$2
  if [ -e "$file_source" ];then
    cp $file_source $file_dest
    return 0
  else
    echo "Error.Copy file fail,file isn't exist. Look log for detail information."  
    return 1
  fi
}

function show_res(){
  local file=$1
  local test_type=$2
  if [ "$test_type" = "obstacle" ];then
    python $SCRIPTS_BASE_DIR/show_report.py $file precision_average recall_average \
    frames label_frames perception_frames
  elif [ "$test_type" = "traffic_light" ];then
    python $SCRIPTS_BASE_DIR/show_report.py $file countdown_time_precision countdown_time_recall labeled_traffic_lights_total match_traffic_lights\
    perception_traffic_lights_total precision_average recall_average\
    total_labeled_frames total_match_frames 
  fi
}

function accumulate(){
  file=$1
  res=$(python $SCRIPTS_BASE_DIR/get_report_res.py $file precision_average recall_average \
    countdown_time_precision countdown_time_recall)
  TRAFFIC_LIGHT_PRECISION=$(echo "$TRAFFIC_LIGHT_PRECISION+`echo $res | awk -F - '{print $1}'`" | bc)
  TRAFFIC_LIGHT_RECALL=$(echo "$TRAFFIC_LIGHT_RECALL+`echo $res | awk -F - '{print $2}'`" | bc)
  COUNTDOWN_PRECISION=$(echo "$COUNTDOWN_PRECISION+`echo $res | awk -F - '{print $3}'`" | bc)
  COUNTDOWN_RECALL=$(echo "$COUNTDOWN_RECALL+`echo $res | awk -F - '{print $4}'`" | bc)
}

function run_test(){
  local test_xml=$1
  local test_type=$2
  if [ "$test_type" = "obstacle" ];then
    REPORT_PATH=$(eval "$BIN_GET_PARAM $CMD_GET_REPORT_PATH ${SCRIPTS_BASE_DIR}/$test_xml")
    REPORT_NAME=$(eval "$BIN_GET_PARAM $CMD_GET_REPORT_NAME ${SCRIPTS_BASE_DIR}/$test_xml")
  elif [ "$test_type" = "traffic_light" ];then
    REPORT_PATH=$(eval "$BIN_GET_PARAM $CMD_TRAFFIC_LIGHT_REPORT_PATH ${SCRIPTS_BASE_DIR}/$test_xml")
    REPORT_NAME=$(eval "$BIN_GET_PARAM $CMD_TRAFFIC_LIGHT_REPORT_NAME ${SCRIPTS_BASE_DIR}/$test_xml")
  fi
  ./test.sh ${test_xml}
  copy_file "${REPORT_PATH}${REPORT_NAME}"  "${REPORT_BACKUP_PATH}/${MASTER_COMMIT_ID}_${REPORT_NAME}"
  if [ "$?" -eq "1" ];then
    backup_log
    exit 1
  fi
  echo "report_path = $REPORT_PATH"
  echo "report_name = $REPORT_NAME"
  echo "==========report result=========="
  show_res $REPORT_PATH/$REPORT_NAME $test_type
  if [ "$test_type" = "traffic_light" ];then
    accumulate $REPORT_PATH/$REPORT_NAME
  fi
}

rm -rf release
RELEASE_PATH=~/.cache/local_roadstar_release/roadstar
ln -s $RELEASE_PATH release

cd release
source ./scripts/update_resources.sh
cd $ROADSTAR_ROOT_DIR

source ./scripts/set_env.sh
export ROS_MASTER_URI='http://localhost:11311'
MASTER_COMMIT_ID=$(git log | head -1 | cut -d ' ' -f 2)
echo "MASTER_COMMIT_ID = $MASTER_COMMIT_ID"

TEST_BASE_DIR="/roadstar/release/modules/integration_test/perception"
SCRIPTS_BASE_DIR="${TEST_BASE_DIR}/scripts"
EXAMPLE_XML_ONE="param/bags_9000_00-03.xml"
EXAMPLE_XML_TEN="param/bags_9000_00-39.xml"
CITY_EXAMPLE_XML_TEN="param/bags_3000_17-56.xml"
BIN_TEST_PREFIX_DIR="/roadstar/release"
BIN_GET_PARAM="/roadstar/release/modules/integration_test/common/xml_param/get_bag_parameter"
CMD_GET_REPORT_NAME="report_name"
CMD_GET_REPORT_PATH="report_path"
REPORT_BACKUP_PATH="/private/zhangzijian/jenkins"
CMD_TRAFFIC_LIGHT_REPORT_PATH="traffic_light_report_path";
CMD_TRAFFIC_LIGHT_REPORT_NAME="traffic_light_report_name";

TRAFFIC_LIGHT_PRECISION="0"
TRAFFIC_LIGHT_RECALL="0"
COUNTDOWN_PRECISION="0"
COUNTDOWN_RECALL="0"

cd $SCRIPTS_BASE_DIR

#run highway test
temp_json='{"commit_id":"'$MASTER_COMMIT_ID'",'
run_test  ${EXAMPLE_XML_TEN} obstacle
echo "[1/3] drive scene highway obstacle INTEGRATION TEST for deep lidar FINISHED"

#check report and record result
REPORT_PATH=$(eval "$BIN_GET_PARAM $CMD_GET_REPORT_PATH ${SCRIPTS_BASE_DIR}/$EXAMPLE_XML_TEN")
REPORT_NAME=$(eval "$BIN_GET_PARAM $CMD_GET_REPORT_NAME ${SCRIPTS_BASE_DIR}/$EXAMPLE_XML_TEN")
if [ ! -e "$REPORT_PATH/$REPORT_NAME" ];then
  temp_json=$temp_json'"highway":{"precision_average":"0","recall_average":"0","velocity_diff_norm_average":"0","velocity_precision_p50":"0","velocity_precision_p95":"0","status":"false"}}'
  upload_integration_result $temp_json
  backup_log
  exit 1
else
  res=$(python $SCRIPTS_BASE_DIR/get_report_res.py $REPORT_PATH/$REPORT_NAME precision_average recall_average velocity_diff_norm_average velocity_precision_p50 velocity_precision_p95)
  t_precision_average=`echo $res | awk -F - '{print $1}'`
  t_recall_average=`echo $res | awk -F - '{print $2}'`
  t_velocity_diff_norm_average=`echo $res | awk -F - '{print $3}'`
  t_velocity_precision_p50=`echo $res | awk -F - '{print $4}'`
  t_velocity_precision_p95=`echo $res | awk -F - '{print $5}'`
  temp_json=$temp_json'"highway":{"precision_average":"'${t_precision_average}'","recall_average":"'${t_recall_average}'","velocity_diff_norm_average":"'${t_velocity_diff_norm_average}'","velocity_precision_p50":"'${t_velocity_precision_p50}'","velocity_precision_p95":"'${t_velocity_precision_p95}'",'
fi
IS_VALID=$(python $SCRIPTS_BASE_DIR/check_report_value.py "$REPORT_PATH/$REPORT_NAME")
if [ "$IS_VALID" -eq "1" ];then
  echo "Warning recall or precision precent is less than standard value. Exit now."
  backup_log
  temp_json=$temp_json'"status":"false"}}'
  upload_integration_result $temp_json
  exit 1
else
  temp_json=$temp_json'"status":"true"},'
fi

#run city test
run_test  ${CITY_EXAMPLE_XML_TEN} obstacle
echo "[2/3] drive scene city obstacle INTEGRATION TEST for deep lidar FINISHED"

#check report and record result
REPORT_PATH=$(eval "$BIN_GET_PARAM $CMD_GET_REPORT_PATH ${SCRIPTS_BASE_DIR}/$CITY_EXAMPLE_XML_TEN")
REPORT_NAME=$(eval "$BIN_GET_PARAM $CMD_GET_REPORT_NAME ${SCRIPTS_BASE_DIR}/$CITY_EXAMPLE_XML_TEN")
if [ ! -e "$REPORT_PATH/$REPORT_NAME" ];then
  temp_json=$temp_json'"city":{"precision_average":"0","recall_average":"0","velocity_diff_norm_average":"0","velocity_precision_p50":"0","velocity_precision_p95":"0","status":"false"}}'
  upload_integration_result $temp_json
  backup_log
  exit 1
else
   res=$(python $SCRIPTS_BASE_DIR/get_report_res.py $REPORT_PATH/$REPORT_NAME precision_average recall_average velocity_diff_norm_average velocity_precision_p50 velocity_precision_p95)
   t_precision_average=`echo $res | awk -F - '{print $1}'`
   t_recall_average=`echo $res | awk -F - '{print $2}'`
   t_velocity_diff_norm_average=`echo $res | awk -F - '{print $3}'`
   t_velocity_precision_p50=`echo $res | awk -F - '{print $4}'`
   t_velocity_precision_p95=`echo $res | awk -F - '{print $5}'`
  temp_json=$temp_json'"city":{"precision_average":"'${t_precision_average}'","recall_average":"'${t_recall_average}'","velocity_diff_norm_average":"'${t_velocity_diff_norm_average}'","velocity_precision_p50":"'${t_velocity_precision_p50}'","velocity_precision_p95":"'${t_velocity_precision_p95}'",'
  if [ $(echo "$t_precision_average < 0.83" | bc) -eq 1 -o $(echo "$t_recall_average < 0.48" | bc) -eq 1 ];then
    echo "Error.city mode integration test result is lower than standard value.Exiting now."
    backup_log
    temp_json=$temp_json'"status":"false"}}'
    upload_integration_result $temp_json
    exit 1
  else
    temp_json=$temp_json'"status":"true"},'
  fi
fi

#run traffic light test
for i in {0..7}
do
  run_test "param/traffic_light_test_${i}.xml" traffic_light
done

#check traffic light report
TRAFFIC_LIGHT_PRECISION=$(awk 'BEGIN{printf "%.2f\n",'${TRAFFIC_LIGHT_PRECISION}'/'8.0'}')
TRAFFIC_LIGHT_RECALL=$(awk 'BEGIN{printf "%.2f\n",'${TRAFFIC_LIGHT_RECALL}'/'8.0'}')
COUNTDOWN_PRECISION=$(awk 'BEGIN{printf "%.2f\n",'${COUNTDOWN_PRECISION}'/'8.0'}')
COUNTDOWN_RECALL=$(awk 'BEGIN{printf "%.2f\n",'${COUNTDOWN_RECALL}'/'8.0'}')

echo "TRAFFIC_LIGHT_PRECISION=$TRAFFIC_LIGHT_PRECISION"
echo "TRAFFIC_LIGHT_RECALL=$TRAFFIC_LIGHT_RECALL"
echo "COUNTDOWN_PRECISION=$COUNTDOWN_PRECISION"
echo "COUNTDOWN_RECALL=$COUNTDOWN_RECALL"

temp_json=$temp_json'"traffic_light":{"traffic_light_precision":"'${TRAFFIC_LIGHT_PRECISION}'","traffic_light_recall":"'${TRAFFIC_LIGHT_RECALL}'","countdown_precision":"'${COUNTDOWN_PRECISION}'","countdown_recall":"'${COUNTDOWN_RECALL}'",'

if [ $(echo "$TRAFFIC_LIGHT_PRECISION < 0.88" | bc) -eq 1 -o $(echo "$TRAFFIC_LIGHT_RECALL < 0.78" | bc) -eq 1 -o $(echo "$COUNTDOWN_PRECISION < 0.59" | bc) -eq 1 -o $(echo "$COUNTDOWN_RECALL < 0.59" | bc) -eq 1 ];then
  echo "Error.Traffic light detection result is lower than standard value.Exiting now."
  backup_log
  temp_json=$temp_json'"status":"false"}}'
  upload_integration_result $temp_json
  exit 1
else
  echo "Traffic light detection test passed."
  temp_json=$temp_json'"status":"true"}}'
  upload_integration_result $temp_json
fi

echo "[3/3] drive scene city traffic light INTEGRATION TEST for deep lidar FINISHED"

echo "Done"
backup_log
