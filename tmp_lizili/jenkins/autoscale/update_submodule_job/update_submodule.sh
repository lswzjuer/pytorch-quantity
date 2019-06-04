#!/usr/bin/env bash

#git pull
cd /roadstar
git config user.email "jenkins@fabu.ai"
git config user.name "Jenkins"

git checkout origin/master -B master
git pull origin master
git submodule init
git submodule update 
git submodule foreach git checkout master
git submodule foreach git pull origin master

# update release refs
RELEASE_VERSION=$(git log --pretty=short origin/release -- | head -n 1 | cut -d ' ' -f 2 | head -c 8) 
echo "RELEASE_VERSION=$RELEASE_VERSION"
cd /private/roadstar-bin/ && \
  rm -f release/roadstar.tar.gz && \
  ln  /private/roadstar-bin/$RELEASE_VERSION/roadstar.tar.gz  release/roadstar.tar.gz && \
  rm -f release/version && \
  ln  /private/roadstar-bin/$RELEASE_VERSION/version  release/version && \
  cd -

cd /roadstar
./roadstar.sh build_fe
if [ $? -eq 0 ]; then
  echo 'roadstar build fe passed!'
else
  echo 'roadstar build fe failed!'
  exit 1
fi
./roadstar.sh build_opt_gpu
if [ $? -eq 0 ]; then
  echo 'roadstar build opt gpu passed!'
else
  echo 'roadstar build opt gpu failed!'
  exit 1
fi


#git push
git commit -am "automatically update submodules commit id Reviewed By jenkins"
git push origin master
