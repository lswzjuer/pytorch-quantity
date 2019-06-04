#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RELEASE_PATH=~/.cache/roadstar_release
RESOURCES_PATH=$HOME/.cache/resources

LOCAL_RELEASE_PATH=$1
echo LOCAL_RELEASE_PATH=$LOCAL_RELEASE_PATH
if [ ! -d "$LOCAL_RELEASE_PATH" ];then 
  echo "LOCAL_RELEASE_PATH doesn't exist.Exiting now."
  exit 1
fi

mkdir -p $RELEASE_PATH

#update roadstar
echo "=============== Pull Roadstar ==============="
rm -f $RELEASE_PATH/roadstar.tar.gz
mv $LOCAL_RELEASE_PATH/roadstar.tar.gz $RELEASE_PATH/
cd $RELEASE_PATH
rm -rf roadstar
tar axf roadstar.tar.gz 
rm -rf roadstar.tar.gz

# modify conf
cd $DIR/../../
rm release
ln -s $RELEASE_PATH/roadstar release

#show new version of release
if [ -e "release/meta.ini" ];then
  RELEASE_VERSION=`cat release/meta.ini | grep git_commit`
  RELEASE_VERSION=${RELEASE_VERSION#* }
  RESOURCE_VERSION=`cat release/meta.ini | grep resource_commit`
  RESOURCE_VERSION=${RESOURCE_VERSION#* }
  echo "=============== After update, current release version: ==============="
  echo "RELEASE_VERSION = ${RELEASE_VERSION}"
fi

# update ros and driver
echo "=============== Pull Ros ==============="
rm -f $RELEASE_PATH/ros.tar.gz
mv $LOCAL_RELEASE_PATH/ros.tar.gz $RELEASE_PATH/
cd $RELEASE_PATH
rm -rf ros
tar axf ros.tar.gz
rm -rf ros.tar.gz
rm -rf ~/.cache/ros/*
cp -rf ros ~/.cache

# add resources
echo "=============== Update resoures ==============="
source $DIR/../../scripts/update_resources.sh
ln -s $RESOURCES_PATH $RELEASE_PATH/roadstar/resources
if [ ! -z "$RESOURCE_VERSION" ]; then
  cd $RESOURCES_PATH
  git checkout $RESOURCE_VERSION
  cd -
fi

cd $RELEASE_PATH/roadstar
echo "release_type: $COMMIT" >> meta.ini
