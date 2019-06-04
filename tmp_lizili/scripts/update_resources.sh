#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ -z $(command -v git-lfs) ]; then
  echo "No git lfs, please update resources in docker"
  exit 1
fi

ssh git@git.fabu.ai
HAS_SSH=`echo $?`
if [ $HAS_SSH == 0 ]; then
  RESOURCES_REMOTE_PATH="git@git.fabu.ai:open/resources.git"
else
  RESOURCES_REMOTE_PATH="http://git.fabu.ai:7070/open/resources.git"
fi
RESOURCES_PATH=$HOME/.cache/resources
RESOURCES_LINK_PATH=$DIR/../resources

if [ ! -d $RESOURCES_PATH ]; then
  git lfs clone $RESOURCES_REMOTE_PATH $RESOURCES_PATH
  cd $RESOURCES_PATH && git lfs install && git lfs checkout
fi
if [ ! -L $RESOURCES_LINK_PATH ]; then
  ln -s $RESOURCES_PATH $RESOURCES_LINK_PATH
fi

cd $RESOURCES_PATH && git lfs install && git remote set-url origin $RESOURCES_REMOTE_PATH && \
  git checkout master && git pull origin master && git lfs pull
