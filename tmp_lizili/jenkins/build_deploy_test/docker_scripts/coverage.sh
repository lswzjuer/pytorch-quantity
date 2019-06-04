#!/usr/bin/env bash

source ./scripts/set_env.sh
export ROS_MASTER_URI='http://localhost:11311'
MASTER_COMMIT_ID=$(git log | head -1 | cut -d ' ' -f 2)
# SUBMODULE_LIST_TEMP=('modules/dreamview')
SUBMODULE_LIST_TEMP=('modules/simulation' 'modules/perception_v2' 'modules/planning' 'modules/hdmap' 'modules/dreamview')
NUM_SUBMOUDLE=${#SUBMODULE_LIST_TEMP[@]}
TEMP='_'
cat $SUBMODULE_LIST_TEMP
all_success="true"
temp_json='{"commit_id":"'$MASTER_COMMIT_ID'","submodules":['
FAIL_TARGET_LIST=""
for ((i=0;i<NUM_SUBMOUDLE;i++)){
    MODULE_NAME=${SUBMODULE_LIST_TEMP[$i]}
    out_put=$(mktemp)
    bash ./scripts/run_coverage.sh //$MODULE_NAME/... $MODULE_NAME > $out_put
    return_code=$?
    fail_target=$(cat $out_put | grep "Failed target:" | cut -b 15- | xargs echo)
    echo "=======out_put begin ========"
    echo "out_put = $(cat $out_put)"
    echo "=======out_put end ========"
    if [ ! -z "$fail_target" ];then
      echo "fail_target = $fail_target"
      FAIL_TARGET_LIST="$fail_target,$FAIL_TARGET_LIST"
    fi
    if [ ${return_code} -eq 0 ];then
        COVERAGE='{"title":"'${MODULE_NAME:8}'","status":"True"}'
    else
        COVERAGE='{"title":"'${MODULE_NAME:8}'","status":"False","fail_target":"'${fail_target}'"}'
        all_success="false"
    fi
    rm $out_put
    if [ $i == 0 ];then
        temp_json=${temp_json}${COVERAGE}
    else
        temp_json=${temp_json},${COVERAGE}
    fi
}
temp_json=${temp_json}']}'
echo $temp_json > /private/zhangzijian/jenkins/coverage_$MASTER_COMMIT_ID.json
echo "coverage_json=$temp_json"
curl -d $temp_json -H "Content-Type: application/json" -X POST 192.168.3.113:5000/api/postUnit

if [ ! -z "$FAIL_TARGET_LIST" ];then
  echo "FAIL_TARGET_LIST=$FAIL_TARGET_LIST"
fi

if [ "$all_success" = "false" ];then
  echo "Run_coverage failed. Exiting now."
  exit 1
fi
