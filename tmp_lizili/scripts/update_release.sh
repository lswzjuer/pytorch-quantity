#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RELEASE_PATH=~/.cache/roadstar_release
RESOURCES_PATH=$HOME/.cache/resources
HEAD_VERSION_PATH=$RELEASE_PATH/master-HEAD
STABLE_VERSION_PATH=$RELEASE_PATH/stable
RANDOM_VERSION_PATH=$RELEASE_PATH/random

if [ ! -f "release/meta.ini" ]; then
  echo "=============== Download new roadstar release ==============" 
else
  RELEASE_VERSION=`cat release/meta.ini`
  RELEASE_VERSION=${RELEASE_VERSION#* }
  LOCAL_HEAD_VERSION=`cat $HEAD_VERSION_PATH/roadstar/meta.ini`
  LOCAL_HEAD_VERSION=${LOCAL_HEAD_VERSION#* }
  LOCAL_HEAD_VERSION=${LOCAL_HEAD_VERSION:0:8}
  LOCAL_STABLE_VERSION=`cat $STABLE_VERSION_PATH/roadstar/meta.ini`
  LOCAL_STABLE_VERSION=${LOCAL_STABLE_VERSION#* }
  LOCAL_STABLE_VERSION=${LOCAL_STABLE_VERSION:0:8}
  LOCAL_RANDOM_VERSION=`cat $RANDOM_VERSION_PATH/roadstar/meta.ini`
  LOCAL_RANDOM_VERSION=${LOCAL_RANDOM_VERSION#* }
  LOCAL_RANDOM_VERSION=${LOCAL_RANDOM_VERSION:0:8}
  echo "=============== Current release version: ==============="
  echo $RELEASE_VERSION
  echo "local HEAD version: "$LOCAL_HEAD_VERSION
  echo "local stable version: "$LOCAL_STABLE_VERSION
  echo "local random version: "$LOCAL_RANDOM_VERSION
fi

# update roadstar release
COMMIT=release
mkdir -p $RELEASE_PATH
mkdir -p $HEAD_VERSION_PATH
mkdir -p $STABLE_VERSION_PATH
mkdir -p $RANDOM_VERSION_PATH
if [ $# == 1 ]; then
  COMMIT=$1
fi

#update roadstar
function Update(){
  echo "=============== Pull Roadstar ==============="
  wget http://release.fabu.ai/$COMMIT/roadstar.tar.gz -P $RELEASE_PATH || exit 1
  echo "=============== Current Release Version Will Be Covered! ==================="
  cd $RELEASE_PATH
  if [ $COMMIT == "master/HEAD" ]; then
     rm -rf $HEAD_VERSION_PATH/roadstar
     tar axf roadstar.tar.gz -C $HEAD_VERSION_PATH
     echo "release_type: $COMMIT" >> $HEAD_VERSION_PATH/roadstar/meta.ini
  elif [ $COMMIT == "release" ]; then
     rm -rf  $STABLE_VERSION_PATH/roadstar
     tar axf roadstar.tar.gz -C $STABLE_VERSION_PATH
     echo "release_type: $COMMIT" >> $STABLE_VERSION_PATH/roadstar/meta.ini
  else
     rm -rf $RANDOM_VERSION_PATH/roadstar
     tar axf roadstar.tar.gz -C $RANDOM_VERSION_PATH
     echo "release_type: random version $COMMIT" >> $RANDOM_VERSION_PATH/roadstar/meta.ini
fi
  rm -rf roadstar.tar.gz
}
#Contrast version
function Contrast(){
  wget http://release.fabu.ai/$COMMIT/version -P $RELEASE_PATH 
  REMOTE_VERSION=`cat $RELEASE_PATH/version`  
  echo "local release version :"$1
  echo "remote release version:"$REMOTE_VERSION
  if [[ -e "$RELEASE_PATH/version" &&  $REMOTE_VERSION == $1 ]]; then
    echo "Release Version "$REMOTE_VERSION" Exist !"
  else
    Update 
  fi
}

rm -rf $RELEASE_PATH/roadstar
rm -f $RELEASE_PATH/roadstar.tar.gz
rm -f $RELEASE_PATH/version

#Local version contrast with remote version
if [ $COMMIT == "master/HEAD" ]; then
  Contrast $LOCAL_HEAD_VERSION
elif [ $COMMIT == "release" ]; then
  Contrast $LOCAL_STABLE_VERSION  
else
  if [ $COMMIT == $LOCAL_RANDOM_VERSION ]; then
    echo "Release Version "$LOCAL_VERSION" Exist !"
  else
    Update
  fi
fi

# modify conf
cd $DIR/..
rm release
if [ $COMMIT == "master/HEAD" ]; then
  ln -s $HEAD_VERSION_PATH/roadstar release
elif [ $COMMIT == "release" ]; then
  ln -s $STABLE_VERSION_PATH/roadstar release
else
  ln -s $RANDOM_VERSION_PATH/roadstar release
fi

ln -s `pwd`/data release/data

#show new version of release
if [ -e "release/meta.ini" ];then
  RELEASE_VERSION=`cat release/meta.ini | grep git_commit`
  RELEASE_VERSION=${RELEASE_VERSION#* }
  RESOURCE_VERSION=`cat release/meta.ini | grep resource_commit`
  RESOURCE_VERSION=${RESOURCE_VERSION#* }
  echo "=============== After update, current release version: ==============="
  echo "RELEASE_VERSION = ${RELEASE_VERSION}"
fi

# add resources
echo "=============== Update resoures ==============="
source $DIR/update_resources.sh
cd $DIR/..
ln -sf $RESOURCES_PATH release/
if [ ! -z "$RESOURCE_VERSION" ]; then
  cd $RESOURCES_PATH
  git checkout $RESOURCE_VERSION
  cd -
fi
