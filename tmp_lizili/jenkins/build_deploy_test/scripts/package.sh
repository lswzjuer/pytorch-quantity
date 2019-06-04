#!/usr/bin/env bash
echo 'package binary'
./roadstar_docker.sh release
RELEASE_PATH=~/.cache/local_roadstar_release/roadstar
COMMIT=`awk '/git_commit/{print $2}' $RELEASE_PATH/meta.ini`
COMMIT=${COMMIT:0:8}
echo 'code version: '$COMMIT
cd $RELEASE_PATH/..
rm -rf $COMMIT
mkdir $COMMIT
tar -czf $COMMIT/roadstar.tar.gz roadstar
tar -czf $COMMIT/ros.tar.gz ros
