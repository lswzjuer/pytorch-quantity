#!/bin/bash

while read line
do
  echo $line
  if [[ "$line" == "#"* ]]; then
    continue;
  fi
  arr=(${line//:/ })
  cd modules/${arr[0]}
  git remote update
  git checkout ${arr[1]} || exit 1
  echo ${arr[1]}
  cd ../..
done < submodules
