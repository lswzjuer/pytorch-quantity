#!/usr/bin/env bash
echo 'deploy binary'
RELEASE_PATH=~/.cache/local_roadstar_release/roadstar
COMMIT=`awk '/git_commit/{print $2}' $RELEASE_PATH/meta.ini`
COMMIT=${COMMIT:0:8}
echo 'code version: '$COMMIT
rm -rf $DEPLOY_PATH/$COMMIT
DIR=`cd $RELEASE_PATH/.. && pwd`  
echo "dir=$DIR"
mv  $DIR/$COMMIT  $DEPLOY_PATH
if [ -z "$DO_NOT_UPDATE_HEAD" ]; then
  rm -rf $DEPLOY_PATH/HEAD
  ln -s $DEPLOY_PATH/$COMMIT $DEPLOY_PATH/HEAD
fi
