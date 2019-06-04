#! /usr/bin/env bash


SCRIPT_DIR=$(cd "$( dirname "${BASH_SOURCE[0]}" )/" && pwd )
source "${SCRIPT_DIR}/../base.sh"

TRANSFORMER="obstacle/transform/label_data_transform"

function help() {
  echo " Usage: 
  ./$0 [PARAMS]"
  echo "To help parse and transform labeled data.Only one param is needed which is the config xml path of the bag."
}

function transform_file(){
  CONF_XML="$CUR_DIR/$1"
  echo " xml = $CONF_XML"
  start $TRANSFORMER  $CONF_XML
  echo -n "begin transforming..."
  for (( ; ; ))
  do
    NUM="$(pgrep -c "label_data_tran")"
    if [ ${NUM} -eq "0" ]; then
      break
    else
      sleep 1
      echo -n "."
    fi
  done
  echo ""
  echo "transform complete! xml = $CONF_XML"
}

function transform_path(){
  FILE_PATH=$1
  for file in $(find $FILE_PATH -name "*.xml")
  do
    echo "file = $file"
    transform_file $file
  done
}
 
function main() {
  local args=$1
  if [ -d "$args" ];then
    echo "$args is path"
    transform_path $args
  elif [ -f "$args" ];then
    echo "$args is file"
    transform_file $args
  fi
}



main $@

