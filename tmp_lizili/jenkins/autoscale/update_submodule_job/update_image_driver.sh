#!/usr/bin/env bash

cd /roadstar
TMP_PATH=".tmp-$(date '+%H%M%S')"
echo "TMP_PATH=$TMP_PATH"
mkdir -p $TMP_PATH/roadstar/scripts
ROADSTAR_SCRIPTS="docker_adduser.sh roadstar_base.sh set_env.sh update_release.sh update_resources.sh"
for file in $ROADSTAR_SCRIPTS
do
  cp scripts/$file $TMP_PATH/roadstar/scripts/
done

mkdir -p $TMP_PATH/roadstar/docker
cp -r docker/scripts $TMP_PATH/roadstar/docker/
cp -r jenkins $TMP_PATH/roadstar
cd $TMP_PATH && tar -czf roadstar.tar.gz roadstar && rm -rf roadstar && cd -

echo "DEPLOY_PATH=$DEPLOY_PATH"
cd $DEPLOY_PATH
mv /roadstar/$TMP_PATH $TMP_PATH
rm -rf image_driver
mv $TMP_PATH image_driver


