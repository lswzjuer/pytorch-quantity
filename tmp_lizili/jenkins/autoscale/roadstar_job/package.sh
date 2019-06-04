#!/usr/bin/env bash
echo 'package binary'
cd /roadstar
./scripts/release.sh release
RELEASE_PATH=~/.cache/local_roadstar_release/roadstar
COMMIT=`awk '/git_commit/{print $2}' $RELEASE_PATH/meta.ini`
COMMIT=${COMMIT:0:8}
echo 'code version: '$COMMIT
cd $RELEASE_PATH/..
rm -rf $COMMIT
mkdir $COMMIT
echo $COMMIT > $COMMIT/version

tar -czf $COMMIT/roadstar.tar.gz roadstar
cp -r $COMMIT ~/local_roadstar_release
