#!/usr/bin/env bash

#git pull
git checkout origin/master -B master
git pull origin master
git submodule init
git submodule update 
git submodule foreach git checkout master
git submodule foreach git pull origin master

# update release refs
RELEASE_VERSION=$(git log --pretty=short origin/release -- | head -n 1 | cut -d ' ' -f 2 | head -c 8) && cd /private/roadstar-bin/ && rm -f release &&  ln -sf /private/roadstar-bin/$RELEASE_VERSION release && cd -

./docker/scripts/dev_start.sh
./roadstar_docker.sh build_fe
./roadstar_docker.sh build_opt_gpu
if [ $? -eq 0 ]; then
  echo 'roadstar build opt gpu passed!'
else
  exit 1
fi

#build
./roadstar_docker.sh build_driver
if [ $? -eq 0 ]; then
  echo 'roadstar build driver passed!'
else
  exit 2
fi

#git push
git commit -am "automatically update submodules commit id Reviewed By jenkins"
git push origin master
