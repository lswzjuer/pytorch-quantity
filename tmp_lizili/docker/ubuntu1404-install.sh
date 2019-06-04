#!/usr/bin/env bash

PACKAGE_PATH=/tmp/package

mkdir -p $PACKAGE_PATH
cp ./requirements.txt $PACKAGE_PATH/
cd $PACKAGE_PATH

function download() {
  rm -rf $PACKAGE_PATH/*
  wget https://github.com/bazelbuild/bazel/releases/download/0.5.4/bazel-0.5.4-installer-linux-x86_64.sh
  wget https://github.com/google/protobuf/releases/download/v3.1.0/protobuf-cpp-3.1.0.tar.gz
  wget https://github.com/google/protobuf/releases/download/v3.1.0/protoc-3.1.0-linux-x86_64.zip
  wget https://github.com/tj/n/archive/v2.1.0.tar.gz
  wget http://www.kvaser.com/software/7330130980754/V5_20_0/linuxcan.tar.gz
  wget https://pypi.python.org/packages/29/72/5c1888c4948a0c7b736d10e0f0f69966e7c0874a660222ed0a2c2c6daa9f/pyproj-1.9.5.1.tar.gz
  wget https://github.com/google/glog/archive/v0.3.5.tar.gz
  wget https://github.com/gflags/gflags/archive/v2.2.0.tar.gz
  wget https://www.baslerweb.com/fp-1496750153/media/downloads/software/pylon_software/pylon-5.0.9.10389-x86_64.tar.gz
  wget https://github.com/opencv/opencv/archive/2.4.13.tar.gz
}

function apt-install() {
  apt-get update && apt-get install -y \
     apt-transport-https \
     bc \
     build-essential \
     cmake \
     cppcheck \
     curl \
     debconf-utils \
     doxygen \
     gdb \
     git \
     lcov \
     libboost-all-dev \
     libcurl4-openssl-dev \
     libfreetype6-dev \
     lsof \
     python-pip \
     python-matplotlib \
     python-scipy \
     python-software-properties \
     realpath \
     software-properties-common \
     unzip \
     vim \
     wget \
     zip
}

function install() {
  # install bazel
  cd $PACKAGE_PATH
  chmod +x bazel-0.5.4-installer-linux-x86_64.sh
  ./bazel-0.5.4-installer-linux-x86_64.sh --prefix=/usr
  
  cd $PACKAGE_PATH
  tar xzf protobuf-cpp-3.1.0.tar.gz
  cd $PACKAGE_PATH/protobuf-3.1.0
  ./configure CXXFLAGS=-fPIC --with-pic=PIC --prefix=/usr && make -j"$(nproc)" && make install
  
  # install protoc 3.1.0
  cd $PACKAGE_PATH
  unzip protoc-3.1.0-linux-x86_64.zip -d protoc3
  mv protoc3/bin/protoc /usr/bin/
  chmod 755 /usr/bin/protoc
  
  # Install yarn
  curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | sudo apt-key add -
  echo "deb https://dl.yarnpkg.com/debian/ stable main" | sudo tee /etc/apt/sources.list.d/yarn.list
  apt-get update && apt-get install -y yarn
  
  # install dependency for ros build
  sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
  apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key 421C365BD9FF1F717815A3895523BAEEB01FA116
  apt-get update && apt-get install -y \
      ros-indigo-catkin \
      libbz2-dev \
      libconsole-bridge-dev \
      liblog4cxx10-dev \
      libeigen3-dev \
      liblz4-dev \
      libpoco-dev \
      libproj-dev \
      libtinyxml-dev \
      libyaml-cpp-dev \
      sip-dev \
      uuid-dev \
      zlib1g-dev \
      libpcap-dev
  
  add-apt-repository "deb http://archive.ubuntu.com/ubuntu trusty-backports universe"
  add-apt-repository ppa:v-launchpad-jochen-sprickerhof-de/pcl
  apt-get update && apt-get install -y shellcheck libpcl-all
  
  # HD map
  apt-get install -y libpugixml-dev
  
  # Canbus
  # install can-utils
  apt-get install -y can-utils
  #install linux can
  cd $PACKAGE_PATH
  tar -zxvf linuxcan.tar.gz
  cd $PACKAGE_PATH/linuxcan/canlib
  make -j"$(nproc)"  && make install
  
  # Simulation
  # install sumo
  add-apt-repository ppa:sumo/stable
  apt-get update && apt-get install -y sumo sumo-tools sumo-doc
  # pyproj
  cd $PACKAGE_PATH
  tar -zxvf pyproj-1.9.5.1.tar.gz
  cd $PACKAGE_PATH/pyproj-1.9.5.1
  python setup.py build && python setup.py install
  
  # install dependency for caffe build
  apt-get install -y \
    libleveldb-dev \
    libsnappy-dev \
    libopencv-dev \
    libhdf5-serial-dev \
    libatlas-base-dev \
    liblmdb-dev
  
  # Install glog
  cd $PACKAGE_PATH
  tar xzf v0.3.5.tar.gz
  cd $PACKAGE_PATH/glog-0.3.5
  ./configure && make -j"$(nproc)"&& make install
  
  # Install gflags
  cd $PACKAGE_PATH
  tar xzf v2.2.0.tar.gz
  cd $PACKAGE_PATH/gflags-2.2.0
  mkdir build
  cd $PACKAGE_PATH/gflags-2.2.0/build
  CXXFLAGS="-fPIC" cmake .. && make -j"$(nproc)" && make install
  
  # install caffe
  #COPY ./third_party/caffe $PACKAGE_PATH/caffe
  #cd $PACKAGE_PATH/caffe
  #make -j"$(nproc)" all
  #cp -r include/caffe /usr/include
  #cp build/lib/libcaffe.a /usr/lib
  #cp build/lib/libcaffe.so.1.0.0-rc3 /usr/lib/
  #ln -s /usr/lib/libcaffe.so.1.0.0-rc3 /usr/lib/libcaffe.so
  
  #install pylon
  cd $PACKAGE_PATH
  tar -zxvf pylon-5.0.9.10389-x86_64.tar.gz
  tar -zxvf pylon-5.0.9.10389-x86_64/pylonSDK-5.0.9.10389-x86_64.tar.gz -C /opt
  
  # install opencv gpu version
  cd $PACKAGE_PATH
  tar -zxvf 2.4.13.tar.gz
  mkdir $PACKAGE_PATH/opencv-2.4.13/build
  cd $PACKAGE_PATH/opencv-2.4.13/build
  cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D WITH_CUDA=ON \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D WITH_CUBLAS=1 \
      -D CUDA_GENERATION=Kepler \
      ..
  
  make -j"$(nproc)" && make install && cd .. && rm -rf build
  echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv2.conf
  ldconfig 
}

function optional_install() {
  # set up node v8.0.0
  cd $PACKAGE_PATH
  tar xzf v2.1.0.tar.gz
  cd $PACKAGE_PATH/n-2.1.0
  make install
  n 8.0.0
  
  apt-get install -y autoconf \
    libproj-dev \
    proj-bin \
    proj-data \
    libtool \
    libgdal1-dev \
    libxerces-c-dev \
    libfox-1.6-0 \
    libfox-1.6-dev
  
}

function install_py() {
  cd $PACKAGE_PATH
  pip install -r requirements.txt
}

# remove tmp/package files
function clear_package() {
  cd $PACKAGE_PATH
  rm -rf $PACKAGE_PATH/*
}

download
apt-install
install
install_py
optional_install

