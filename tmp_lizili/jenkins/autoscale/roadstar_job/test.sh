#!/bin/zsh

SCRIPT_DIR=$(cd $( dirname "{BASH_SOURCE[0]}" ) && pwd )
cd $SCRIPT_DIR

source ~/.zshrc; nohup roscore </dev/null >'/tmp/roscore.log' 2>&1 &
bash ./jenkins/build_deploy_test/docker_scripts/perception_report.sh 
if [ "$?" -ne "0" ];then
  exit 1
fi

bash ./jenkins/build_deploy_test/docker_scripts/coverage.sh

