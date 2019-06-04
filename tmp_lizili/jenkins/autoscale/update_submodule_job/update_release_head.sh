#!/usr/bin/env bash


# get roadstar log
cd /roadstar
COMMIT_IDS=$(git log --oneline -n 50 | awk '{print $1}' | xargs ) 

echo "COMMIT_IDS=$COMMIT_IDS"

DEPLOY_PATH="/private/roadstar-bin/"
cd $DEPLOY_PATH
for id in $COMMIT_IDS
do
  ret=$(ls | grep "${id}*")
  if [ ! -z "$ret" ];then
    if [ -f "$ret/roadstar.tar.gz" ];then
      echo "find latest ret=$ret under $DEPLOY_PATH and update release HEAD now."
      rm  $DEPLOY_PATH/HEAD/roadstar.tar.gz
      ln  $DEPLOY_PATH/$ret/roadstar.tar.gz $DEPLOY_PATH/HEAD/roadstar.tar.gz
      if [ -e "$DEPLOY_PATH/HEAD/version" ];then
        rm  $DEPLOY_PATH/HEAD/version
      fi
      if [ -e "$DEPLOY_PATH/$ret/version" ];then
        ln  $DEPLOY_PATH/$ret/version $DEPLOY_PATH/HEAD/version
      fi
      break
    fi
  fi
done


