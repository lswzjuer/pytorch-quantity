#!/usr/bin/env bash
echo 'Play bag for test'
function docker_cmd() {
  docker exec -u $USER ${USER}_roadstar_dev bash -c "$@"
}
echo "Port map:"
docker port ${USER}_roadstar_dev
docker exec -u $USER ${USER}_roadstar_dev zsh -c "source ~/.zshrc; nohup roscore </dev/null >'/tmp/roscore.log' 2>&1 &"
docker exec -u $USER ${USER}_roadstar_dev bash ./jenkins/build_deploy_test/docker_scripts/perception_report.sh
if [ "$?" -ne "0" ];then
  exit 1
fi

docker exec -u $USER ${USER}_roadstar_dev bash ./jenkins/build_deploy_test/docker_scripts/coverage.sh

#echo "Bag path: "$BAG_PATH
#docker_cmd "scripts/integration_test.sh $BAG_PATH"
