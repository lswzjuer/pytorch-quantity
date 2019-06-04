#!/usr/bin/env bash

cd /roadstar
mkdir -p docs && doxygen roadstar.doxygen

echo "DEPLOY_PATH=$DEPLOY_PATH"
cd $DEPLOY_PATH

TMP_PATH=".tmp-$(date '+%H%M%S')"
echo "TMP_PATH=$TMP_PATH"
mv /roadstar/docs $TMP_PATH
rm -rf docs
mv $TMP_PATH/doxygen/html docs
rm -rf $TMP_PATH

