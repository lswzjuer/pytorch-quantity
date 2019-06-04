#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${DIR}/roadstar_base.sh"
TOOL_DIR=${ROADSTAR_BIN_PREFIX}/modules/tools/message_service

function help() {
  echo "-----------------------------------------------------------"
  echo "Usage: ./scripts/filter_msg.sh indir outdir types"
  echo "Example:"
  echo "./scripts/filter_msg.sh /onboard_data/bags/jiaxing_data/truck01/20190516/0845 \\"
  echo "                        /onboard_data/bags/jiaxing_data/truck01/20190516/0845/filtered \\"
  echo "                        LOCALIZATION,FUSION_MAP"
  echo "-----------------------------------------------------------"
}

function main() {
  if [ -z $1 ] || [ -z $2 ]; then
    help
    exit 1
  fi
  if [ ! -d $2 ]; then
    mkdir -p $2
  fi
  IN_DIR=`readlink -f $1`
  OUT_DIR=`readlink -f $2`
  TYPES="LOCALIZATION,TRAFFIC_LIGHT_DETECTION,PLANNING_TRAJECTORY,FUSION_MAP,CHASSIS,\
    CONTROL_COMMAND,CONTROL_DEBUG,CONTROL_STATUS,MESSAGE_SERVICE_STATUS,SYSTEM_STATUS"
  TYPES=`echo $TYPES | sed 's/ //g'`
  if [ ! -z $3 ]; then
    TYPES=$3
  fi

  MSGS=`ls $IN_DIR/*.msg`
  MSGNUM=`ls $IN_DIR/*.msg| wc -l`
  echo "found $MSGNUM .msg in input dir"
  for MSG in $MSGS; do
    OUTFILE=$OUT_DIR"/"`basename $MSG`
    echo $MSG "->" $OUTFILE
    $TOOL_DIR/extract_msg -in_file=$MSG -out_file=$OUTFILE -types=$TYPES
  done
}

main $@
