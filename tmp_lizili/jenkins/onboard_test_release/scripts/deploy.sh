#!/usr/bin/env bash
echo 'deploy binary'
RELEASE_PATH=~/.cache/local_roadstar_release/roadstar
COMMIT=`awk '/git_commit/{print $2}' $RELEASE_PATH/meta.ini`
COMMIT=${COMMIT:0:8}
echo 'code version: '$COMMIT
rm -rf $DEPLOY_PATH/$COMMIT
rm -rf $DEPLOY_PATH/HEAD
mkdir -p $DEPLOY_PATH/$COMMIT
cd $RELEASE_PATH/..
tar -czf $DEPLOY_PATH/$COMMIT/roadstar.tar.gz roadstar
tar -czf $DEPLOY_PATH/$COMMIT/ros.tar.gz ros
ln -s $DEPLOY_PATH/$COMMIT $DEPLOY_PATH/HEAD
