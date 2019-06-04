FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

ENV DEBIAN_FRONTEND=noninteractive

RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y \
   apt-transport-https \
   bc \
   build-essential \
   cmake \
   cppcheck \
   curl \
   debconf-utils \
   gdb \
   git \
   lcov \
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
   wget \
   zip \
   zsh \
   ssh \
   net-tools \
   sudo \
   libgtest-dev \
   python-empy \
   python-nose \
   locales\
   ctags\
   libboost-all-dev \
   xterm \
   tmux # vim-nox \
   htop && \
   apt-get clean autoclean && \
   rm -rf /var/lib/apt/lists/*


RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    sed -i 's/http:\/\/ppa.launchpad.net/https:\/\/launchpad.proxy.ustclug.org/g' /etc/apt/sources.list.d/ubuntu-toolchain-r-ubuntu-test-*.list && \
    apt-get update && apt-get install -y gcc-8 g++-8 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 60 \
                        --slave /usr/bin/g++ g++ /usr/bin/g++-8 \
                        --slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-8 \
                        --slave /usr/bin/gcc-nm gcc-nm /usr/bin/gcc-nm-8 \
                        --slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-8 \
                        --slave /usr/bin/gcov gcov /usr/bin/gcov-8 && \
    apt-get clean autoclean && rm -rf /var/lib/apt/lists/*

# install gstreamer
RUN apt-get update && apt-get install -y gstreamer1.0-tools gstreamer1.0-alsa gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-good1.0-dev libgstreamer-plugins-bad1.0-dev \
   && apt-get clean autoclean && rm -rf /var/lib/apt/lists/*

# oping
RUN apt-get update && apt-get install -y inetutils-ping liboping0 liboping-dev oping \
   && apt-get clean autoclean && rm -rf /var/lib/apt/lists/*

# Install required python packages.
WORKDIR /tmp
RUN wget http://release.fabu.ai/deps/py27_requirements.txt && pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ -r py27_requirements.txt && rm -rf /root/.cache/ /tmp/*

# set up node v10
RUN curl -sL http://release.fabu.ai/deps/node_setup.sh | bash - && \
    apt-get install -y nodejs && \
    apt-get clean autoclean && rm -rf /var/lib/apt/lists/*


# Install yarn
RUN curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add - \
  && echo "deb https://dl.yarnpkg.com/debian/ stable main" | tee /etc/apt/sources.list.d/yarn.list \
  && apt-get update && apt-get install -y yarn \
  && apt-get clean autoclean && rm -rf /var/lib/apt/lists/*

ENV ROSCONSOLE_FORMAT '${file}:${line} ${function}() [${severity}] [${time}]: ${message}'

# install dependency for ros build
# RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
# RUN apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key 421C365BD9FF1F717815A3895523BAEEB01FA116

# ros-indigo-catkin \
RUN apt-get update && apt-get install -y \
    libbz2-dev \
    libconsole-bridge-dev \
    liblog4cxx10-dev \
    liblz4-dev \
    libpoco-dev \
    libproj-dev \
    libtinyxml-dev \
    libyaml-cpp-dev \
    sip-dev \
    uuid-dev \
    zlib1g-dev \
    libpcap-dev \
    && apt-get clean autoclean && rm -rf /var/lib/apt/lists/*

# https://stackoverflow.com/questions/25193161/chfn-pam-system-error-intermittently-in-docker-hub-builds
RUN ln -s -f /bin/true /usr/bin/chfn


# HD map
RUN apt-get update && apt-get install -y libpugixml-dev \
    && apt-get clean autoclean && rm -rf /var/lib/apt/lists/*

# Vehile
WORKDIR /tmp
RUN git clone https://github.com/ninja-build/ninja.git && cd ninja && \
    ./configure.py --bootstrap && \
    mv ninja /usr/bin && rm -rf /tmp/*

# install pip3 meson
RUN apt-get update && apt-get install -y  python3-pip \
    && apt-get clean autoclean && rm -rf /var/lib/apt/lists/*

RUN pip3 install meson==0.44 && rm -rf /root/.cache/

# install jsoncpp
WORKDIR /tmp
RUN git clone https://github.com/open-source-parsers/jsoncpp.git && cd jsoncpp && \
    meson --buildtype release --default-library shared . build-shared && \
    ninja -v -C build-shared test && cd build-shared && \
    ninja install && rm -rf /tmp/*

# Simulation
# pyproj
WORKDIR /tmp
RUN wget http://release.fabu.ai/deps/pyproj-1.9.5.1.tar.gz && \
    tar -zxvf pyproj-1.9.5.1.tar.gz && \
    cd /tmp/pyproj-1.9.5.1 && \
    python setup.py build && \
    python setup.py install && \
    cd / && rm -rf /tmp/*

# Canbus
# install can-utils
RUN apt-get update && apt-get install -y can-utils \
    && apt-get clean autoclean && rm -rf /var/lib/apt/lists/*

#install linux can
WORKDIR /tmp
RUN wget http://release.fabu.ai/deps/linuxcan.tar.gz && tar -zxvf linuxcan.tar.gz && \
    cd /tmp/linuxcan/canlib && \
    make -j"$(nproc)" && \
    make install && \
    cd / && rm -rf /tmp/*


# Install glog
WORKDIR /tmp
RUN wget http://release.fabu.ai/deps/glog-0.3.5.tar.gz && \
    tar xzf glog-0.3.5.tar.gz && \
    cd /tmp/glog-0.3.5 && \
    ./configure && \
    make -j"$(nproc)" && \
    make install && \
    cd / && rm -rf /tmp/*

# Install gflags
WORKDIR /tmp
RUN wget http://release.fabu.ai/deps/gflags-2.2.0.tar.gz && \
    tar xzf gflags-2.2.0.tar.gz && \
    cd /tmp/gflags-2.2.0 && \
    mkdir build && \
    cd /tmp/gflags-2.2.0/build && \
    CXXFLAGS="-fPIC" cmake .. && \
    make -j"$(nproc)" && \
    make install && \
    cd / && rm -rf /tmp/*

RUN apt-get update && apt-get install -y autoconf \
  libproj-dev \
  proj-bin \
  proj-data \
  libtool \
  libgdal1-dev \
  libxerces-c-dev \
  libfox-1.6-0 \
  libfox-1.6-dev \
  && apt-get clean autoclean && rm -rf /var/lib/apt/lists/*

#install pylon
WORKDIR /tmp
RUN wget http://release.fabu.ai/deps/pylon-5.1.0.12682-x86_64.tar.gz && \
    tar -zxvf pylon-5.1.0.12682-x86_64.tar.gz && \
    tar -zxvf pylon-5.1.0.12682-x86_64/pylonSDK-5.1.0.12682-x86_64.tar.gz -C /opt && \
    echo '/opt/pylon5/lib64' >> /etc/ld.so.conf.d/pylon.conf && ldconfig && \
    cd / && rm -rf /tmp/*


RUN locale-gen en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
    
# modify ssh port
RUN sed -i 's/Port 22/Port 2222/g' /etc/ssh/sshd_config

WORKDIR /etc
RUN rm -rf /etc/skel && \
    wget http://release.fabu.ai/deps/skel.tar.gz && \
    tar -axf skel.tar.gz && \
    rm -rf skel.tar.gz && \
    rm -rf /tmp/*

# arc
RUN apt-get update && apt-get install -y php7.0 php7.0-curl && \
    mkdir /usr/local/arc && cd /usr/local/arc && \
    git clone http://git.fabu.ai:7070/third_party/arcanist.git && \
    git clone https://github.com/phacility/libphutil.git && \
    ln -s /usr/local/arc/arcanist/bin/arc /usr/bin/arc && \
    apt-get clean autoclean && rm -rf /var/lib/apt/lists/*

# cpplint
WORKDIR /tmp
RUN wget http://release.fabu.ai/deps/cpplint.py && \
cp cpplint.py /usr/bin && \
chmod a+x /usr/bin/cpplint.py && rm -rf /tmp/*

# install deps for opencv
# python-numpy \
RUN apt-get update && apt-get install -y \
      python-dev \
      libtbb2 \
      libtbb-dev \
      libjpeg-dev \
      libpng-dev \
      libtiff-dev \
      libjasper-dev \
      libdc1394-22-dev \
      libgtk2.0-dev && \
      apt-get clean autoclean && rm -rf /var/lib/apt/lists/*


# install deps for pcl
RUN apt-get update && \
    apt-get install -y libflann-dev \
       libvtk6-dev \
       libqhull-dev && \
      apt-get clean autoclean && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
# RUN wget http://release.fabu.ai/deps/opencv-3.3.1.tar.gz -O /tmp/opencv-3.3.1.tar.gz && \
#     wget http://release.fabu.ai/deps/opencv_contrib-3.3.1.tar.gz && \
#     tar -xvf opencv-3.3.1.tar.gz && \
#     tar -xvf opencv_contrib-3.3.1.tar.gz && \
#     mkdir /tmp/opencv-3.3.1/build && \
#     cd /tmp/opencv-3.3.1/build && \
#     cmake -D CMAKE_BUILD_TYPE=RELEASE \
#       -D CMAKE_INSTALL_PREFIX=/usr/local/opencv-3.3.1 \
#       -D WITH_CUDA=ON \
#       -D GSTREAMER=ON \
#       -D ENABLE_FAST_MATH=1 \
#       -D CUDA_FAST_MATH=1 \
#       -D WITH_CUBLAS=1 \
#       -D WITH_GTK=ON \
#       -D CUDA_ARCH_BIN="35 50 60 61 70 75" \
#       -D OPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib-3.3.1/modules \
#       .. && \
#     make -j"$(nproc)" && \
#     make install -j"$(nproc)" && \
#     cd / && rm -rf /tmp/*

RUN mkdir -p /usr/local/opencv-3.3.1 && wget -O- http://release.fabu.ai/deps/opencv-3.3.1.tar.gz | tar -zxvf - --strip-components=1 -C /usr/local/opencv-3.3.1 && echo /usr/local/opencv-3.3.1/lib > /etc/ld.so.conf.d/opencv-3.3.1.conf && ldconfig

# eigen keep same with tensorflow-1.10.0
WORKDIR /tmp
RUN wget http://release.fabu.ai/deps/eigen-eigen-fd6845384b86.tar.gz && \
    tar -zxvf eigen-eigen-fd6845384b86.tar.gz && \
    cd  eigen-eigen-fd6845384b86 && \
    mkdir build && cd build && cmake -DCMAKE_INSTALL_PREFIX=/usr .. && \
    make && make install && \
    cd / && rm -rf /tmp/*

# install pcl
WORKDIR /tmp
RUN wget http://release.fabu.ai/deps/pcl-pcl-1.9.1.tar.gz && tar -zxf pcl-pcl-1.9.1.tar.gz && \
    cd pcl-pcl-1.9.1 && \
    cmake -DCMAKE_BUILD_TYPE=Release . && \
    make -j"$(nproc)" && \
    make install && \
    cd / && rm -rf /tmp/*

# install bazel
WORKDIR /tmp
RUN wget http://release.fabu.ai/deps/bazel-0.17.1-installer-linux-x86_64.sh && \
    chmod +x bazel-0.17.1-installer-linux-x86_64.sh && \
    ./bazel-0.17.1-installer-linux-x86_64.sh --prefix=/usr && \
    cd / && rm -rf /tmp/*

# install protobuf 3.6.0
WORKDIR /tmp
RUN wget http://release.fabu.ai/deps/protobuf-cpp-3.6.0.tar.gz && \
    tar xzf protobuf-cpp-3.6.0.tar.gz && \
    cd /tmp/protobuf-3.6.0 && \
    ./configure CXXFLAGS=-fPIC --with-pic=PIC --prefix=/usr && \
    make -j"$(nproc)" && \
    make install && \
    cd / && rm -rf /tmp/*

# install protoc 3.6.0
WORKDIR /tmp
RUN wget http://release.fabu.ai/deps/protoc-3.6.0-linux-x86_64.zip && \
    unzip protoc-3.6.0-linux-x86_64.zip -d protoc3 && \
    mv protoc3/bin/protoc /usr/bin/ && \
    chmod 755 /usr/bin/protoc && \
    cd / && rm -rf /tmp/*


# install dependency for caffe build
RUN apt-get update && apt-get install -y \
  libleveldb-dev \
  libsnappy-dev \
  libhdf5-serial-dev \
  libatlas-base-dev \
  liblmdb-dev && \
  apt-get clean autoclean && rm -rf /var/lib/apt/lists/*

# install sumo
RUN add-apt-repository ppa:sumo/stable && \
    sed -i 's/http:\/\/ppa.launchpad.net/https:\/\/launchpad.proxy.ustclug.org/g' /etc/apt/sources.list.d/sumo-ubuntu-stable-xenial.list && \
    apt-get update && apt-get install -y sumo sumo-tools sumo-doc \
    && apt-get clean autoclean && rm -rf /var/lib/apt/lists/*

# git lfs
WORKDIR /tmp
RUN wget http://release.fabu.ai/deps/git-lfs-linux-amd64-v2.6.1.tar.gz && \
    tar -zxvf git-lfs-linux-amd64-v2.6.1.tar.gz &&  bash ./install.sh && cd / && rm -rf /tmp/*

# install tcmalloc
RUN apt-get update && apt-get install -y google-perftools \
    && apt-get clean autoclean && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
        libxau6 \
        libxdmcp6 \
        libxcb1 \
        libxext6 \
        libx11-6 \
        ca-certificates \
        automake \
        autoconf \
        libtool \
        pkg-config \
        libxext-dev \
        libx11-dev \
        x11proto-gl-dev && \
    rm -rf /var/lib/apt/lists/*

# install tensorflow
WORKDIR /tmp
RUN wget http://release.fabu.ai/deps/tensorflow-1.10.1.tar.gz && \
    tar -zxvf tensorflow-1.10.1.tar.gz && \
    mkdir -p /usr/local/tensorflow && \
    cd tensorflow && \
    mv include lib /usr/local/tensorflow && \
    rm -rf /tmp/*

# install mxnet
WORKDIR /tmp
RUN wget http://release.fabu.ai/deps/mxnet-1.4.0.tar.gz && \
    tar -zxvf mxnet-1.4.0.tar.gz && \
    mkdir -p /usr/local/mxnet && \
    cd mxnet && \
    mv include lib /usr/local/mxnet && \
    rm -rf /tmp/*

# install mklkl, openblas and lapack
WORKDIR /tmp
# origin download path: https://github.com/intel/mkl-dnn/releases/download/v0.17.2/mklml_lnx_2019.0.1.20180928.tgz 
RUN wget http://release.fabu.ai/deps/mklml_lnx_2019.0.1.20180928.tgz && \
    tar xzvf mklml_lnx_2019.0.1.20180928.tgz && \
    cp mklml_lnx_2019.0.1.20180928/lib/* /usr/local/lib/ && \
    cp mklml_lnx_2019.0.1.20180928/include/* /usr/local/include/ && \
    cd / && rm -rf /tmp/*

RUN apt-get update && \
    apt-get install -y liblapack-dev \
    libblas-dev \
    libopenblas-dev \
    && apt-get clean autoclean && rm -rf /var/lib/apt/lists/*

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
        ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
        ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,compat32,utility

# Required for non-glvnd setups.
ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

WORKDIR /tmp
RUN git clone --branch="0.1.1" https://github.com/NVIDIA/libglvnd.git && cd libglvnd && \
    ./autogen.sh && \
    ./configure --prefix=/usr/local --libdir=/usr/local/lib/x86_64-linux-gnu && \
    make -j"$(nproc)" install-strip && \
    find /usr/local/lib/x86_64-linux-gnu -type f -name 'lib*.la' -delete && \
    rm -rf /tmp/*

RUN mkdir -p /usr/local/share/glvnd/egl_vendor.d/ && wget -O /usr/local/share/glvnd/egl_vendor.d/10_nvidia.json http://release.fabu.ai/deps/10_nvidia.json

RUN echo '/usr/local/lib/x86_64-linux-gnu' >> /etc/ld.so.conf.d/glvnd.conf && \
    ldconfig

ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES},display

# install TensorRT
WORKDIR /tmp
RUN wget http://release.fabu.ai/deps/TensorRT-5.0.2.6.tar.gz && \
    tar -zxvf TensorRT-5.0.2.6.tar.gz && \
    mkdir -p /usr/local/tensorrt && \
    cd TensorRT-5.0.2.6 && \
    mv include lib /usr/local/tensorrt && \
    rm -rf /tmp/*

# install caffe
WORKDIR /tmp
RUN wget http://release.fabu.ai/deps/caffe_cuda10_cudnn7.tar.gz && \
    tar -zxvf caffe_cuda10_cudnn7.tar.gz && \
    mkdir -p /usr/local/caffe && \
    cd caffe_cuda10_cudnn7 && \
    mv include lib /usr/local/caffe && \
    rm -rf /tmp/*

# install llvm
RUN echo 'deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial main' >> /etc/apt/sources.list.d/llvm-toolchain-xenial.list && \
    echo 'deb-src http://apt.llvm.org/xenial/ llvm-toolchain-xenial main' >> /etc/apt/sources.list.d/llvm-toolchain-xenial.list && \
    wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key|sudo apt-key add - && \
    apt-get update && apt-get install -y clang-9 lldb-9 lld-9 clang-tools-9 clang-format-9 clang-tidy-9 libunwind-dev && \
    update-alternatives --install /usr/bin/clang clang /usr/bin/clang-9 60 \
                        --slave /usr/bin/clang++ clang++ /usr/bin/clang++-9 \
                        --slave /usr/bin/clang-apply-replacements clang-apply-replacements /usr/bin/clang-apply-replacements-9 \
                        --slave /usr/bin/clang-change-namespace clang-change-namespace /usr/bin/clang-change-namespace-9 \
                        --slave /usr/bin/clang-check clang-check /usr/bin/clang-check-9 \
                        --slave /usr/bin/clang-cl clang-cl /usr/bin/clang-cl-9 \
                        --slave /usr/bin/clang-cpp clang-cpp /usr/bin/clang-cpp-9 \
                        --slave /usr/bin/clangd clangd /usr/bin/clangd-9 \
                        --slave /usr/bin/clang-tidy clang-tidy /usr/bin/clang-tidy-9 \
                        --slave /usr/bin/clang-format clang-format /usr/bin/clang-format-9 \
                        --slave /usr/bin/clang-format-diff clang-format-diff /usr/bin/clang-format-diff-9 \
                        --slave /usr/bin/clang-func-mapping clang-func-mapping /usr/bin/clang-func-mapping-9 \
                        --slave /usr/bin/clang-import-test clang-import-test /usr/bin/clang-import-test-9 \
                        --slave /usr/bin/clang-include-fixer clang-include-fixer /usr/bin/clang-include-fixer-9 \
                        --slave /usr/bin/clang-offload-bundler clang-offload-bundler /usr/bin/clang-offload-bundler-9 \
                        --slave /usr/bin/clang-query clang-query /usr/bin/clang-query-9 \
                        --slave /usr/bin/clang-refactor clang-refactor /usr/bin/clang-refactor-9 \
                        --slave /usr/bin/clang-rename clang-rename /usr/bin/clang-rename-9 \
                        --slave /usr/bin/ld64.lld ld64.lld /usr/bin/ld64.lld-9 \
                        --slave /usr/bin/ld.lld ld.lld /usr/bin/ld.lld-9 \
                        --slave /usr/bin/lld lld /usr/bin/lld-9 \
                        --slave /usr/bin/lld-link lld-link /usr/bin/lld-link-9  \
                        --slave /usr/bin/wasm-ld wasm-ld /usr/bin/wasm-ld-9 \
                        --slave /usr/bin/lldb lldb /usr/bin/lldb-9 \
                        --slave /usr/bin/lldb-argdumper lldb-argdumper /usr/bin/lldb-argdumper-9 \
                        --slave /usr/bin/lldb-mi lldb-mi /usr/bin/lldb-mi-9 \
                        --slave /usr/bin/lldb-server lldb-server /usr/bin/lldb-server-9 \
                        --slave /usr/bin/lldb-test lldb-test /usr/bin/lldb-test-9 \
                        --slave /usr/bin/bugpoint bugpoint /usr/bin/bugpoint-9 \
                        --slave /usr/bin/dsymutil dsymutil /usr/bin/dsymutil-9 \
                        --slave /usr/bin/llc llc /usr/bin/llc-9 \
                        --slave /usr/bin/llvm-ar llvm-ar /usr/bin/llvm-ar-9 \
                        --slave /usr/bin/llvm-as llvm-as /usr/bin/llvm-as-9 \
                        --slave /usr/bin/llvm-bcanalyzer llvm-bcanalyzer /usr/bin/llvm-bcanalyzer-9 \
                        --slave /usr/bin/llvm-cat llvm-cat /usr/bin/llvm-cat-9 \
                        --slave /usr/bin/llvm-cfi-verify llvm-cfi-verify /usr/bin/llvm-cfi-verify-9 \
                        --slave /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-9 \
                        --slave /usr/bin/llvm-cov llvm-cov /usr/bin/llvm-cov-9 \
                        --slave /usr/bin/llvm-c-test llvm-c-test /usr/bin/llvm-c-test-9 \
                        --slave /usr/bin/llvm-cvtres llvm-cvtres /usr/bin/llvm-cvtres-9 \
                        --slave /usr/bin/llvm-cxxdump llvm-cxxdump /usr/bin/llvm-cxxdump-9 \
                        --slave /usr/bin/llvm-cxxfilt llvm-cxxfilt /usr/bin/llvm-cxxfilt-9 \
                        --slave /usr/bin/llvm-diff llvm-diff /usr/bin/llvm-diff-9 \
                        --slave /usr/bin/llvm-dis llvm-dis /usr/bin/llvm-dis-9 \
                        --slave /usr/bin/llvm-dlltool llvm-dlltool /usr/bin/llvm-dlltool-9 \
                        --slave /usr/bin/llvm-dwarfdump llvm-dwarfdump /usr/bin/llvm-dwarfdump-9 \
                        --slave /usr/bin/llvm-dwp llvm-dwp /usr/bin/llvm-dwp-9 \
                        --slave /usr/bin/llvm-exegesis llvm-exegesis /usr/bin/llvm-exegesis-9 \
                        --slave /usr/bin/llvm-extract llvm-extract /usr/bin/llvm-extract-9 \
                        --slave /usr/bin/llvm-lib llvm-lib /usr/bin/llvm-lib-9 \
                        --slave /usr/bin/llvm-link llvm-link /usr/bin/llvm-link-9 \
                        --slave /usr/bin/llvm-lto2 llvm-lto2 /usr/bin/llvm-lto2-9 \
                        --slave /usr/bin/llvm-lto llvm-lto /usr/bin/llvm-lto-9 \
                        --slave /usr/bin/llvm-mc llvm-mc /usr/bin/llvm-mc-9 \
                        --slave /usr/bin/llvm-mca llvm-mca /usr/bin/llvm-mca-9 \
                        --slave /usr/bin/llvm-modextract llvm-modextract /usr/bin/llvm-modextract-9 \
                        --slave /usr/bin/llvm-mt llvm-mt /usr/bin/llvm-mt-9 \
                        --slave /usr/bin/llvm-nm llvm-nm /usr/bin/llvm-nm-9 \
                        --slave /usr/bin/llvm-objcopy llvm-objcopy /usr/bin/llvm-objcopy-9 \
                        --slave /usr/bin/llvm-objdump llvm-objdump /usr/bin/llvm-objdump-9 \
                        --slave /usr/bin/llvm-opt-report llvm-opt-report /usr/bin/llvm-opt-report-9 \
                        --slave /usr/bin/llvm-pdbutil llvm-pdbutil /usr/bin/llvm-pdbutil-9 \
                        --slave /usr/bin/llvm-PerfectShuffle llvm-PerfectShuffle /usr/bin/llvm-PerfectShuffle-9 \
                        --slave /usr/bin/llvm-profdata llvm-profdata /usr/bin/llvm-profdata-9 \
                        --slave /usr/bin/llvm-ranlib llvm-ranlib /usr/bin/llvm-ranlib-9 \
                        --slave /usr/bin/llvm-rc llvm-rc /usr/bin/llvm-rc-9 \
                        --slave /usr/bin/llvm-readelf llvm-readelf /usr/bin/llvm-readelf-9 \
                        --slave /usr/bin/llvm-readobj llvm-readobj /usr/bin/llvm-readobj-9 \
                        --slave /usr/bin/llvm-rtdyld llvm-rtdyld /usr/bin/llvm-rtdyld-9 \
                        --slave /usr/bin/llvm-size llvm-size /usr/bin/llvm-size-9 \
                        --slave /usr/bin/llvm-split llvm-split /usr/bin/llvm-split-9 \
                        --slave /usr/bin/llvm-stress llvm-stress /usr/bin/llvm-stress-9 \
                        --slave /usr/bin/llvm-strings llvm-strings /usr/bin/llvm-strings-9 \
                        --slave /usr/bin/llvm-strip llvm-strip /usr/bin/llvm-strip-9 \
                        --slave /usr/bin/llvm-symbolizer llvm-symbolizer /usr/bin/llvm-symbolizer-9 \
                        --slave /usr/bin/llvm-tblgen llvm-tblgen /usr/bin/llvm-tblgen-9 \
                        --slave /usr/bin/llvm-undname llvm-undname /usr/bin/llvm-undname-9 \
                        --slave /usr/bin/llvm-xray llvm-xray /usr/bin/llvm-xray-9 \
                        --slave /usr/bin/obj2yaml obj2yaml /usr/bin/obj2yaml-9 \
                        --slave /usr/bin/opt opt /usr/bin/opt-9 \
                        --slave /usr/bin/sanstats sanstats /usr/bin/sanstats-9 \
                        --slave /usr/bin/verify-uselistorder verify-uselistorder /usr/bin/verify-uselistorder-9 \
                        --slave /usr/bin/yaml2obj yaml2obj /usr/bin/yaml2obj-9 \
                        --slave /usr/lib/llvm llvm /usr/lib/llvm-9 && \
      apt-get clean autoclean && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository -y ppa:neovim-ppa/stable && \
    sed -i 's/http:\/\/ppa.launchpad.net/https:\/\/launchpad.proxy.ustclug.org/g' /etc/apt/sources.list.d/neovim-ppa-ubuntu-stable-xenial.list && \
    apt-get update && \
    apt-get install -y neovim && \
    update-alternatives --install /usr/bin/vi vi /usr/bin/nvim 60 && \
    update-alternatives --install /usr/bin/vim vim /usr/bin/nvim 60 && \
    update-alternatives --install /usr/bin/editor editor /usr/bin/nvim 60 && \
    pip3 install pynvim && rm -rf /root/.cache/ && \
    apt-get clean autoclean && \
    rm -rf /var/lib/apt/lists/*

RUN wget -O /usr/local/bin/buildifier http://release.fabu.ai/deps/buildifier && chmod +x /usr/local/bin/buildifier && pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple/ autopep8 && rm -rf /root/.cache/

RUN wget -O /tmp/fd.deb http://release.fabu.ai/deps/fd-musl_7.3.0_amd64.deb && \
    dpkg -i /tmp/fd.deb && \
    apt-get update && \
    apt-get install -y silversearcher-ag && \
    apt-get clean autoclean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /tmp/*

RUN apt-get update && \
    apt-get install -y udev && \
    apt-get clean autoclean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /tmp/*

RUN wget -O /usr/bin/doxygen http://release.fabu.ai/deps/doxygen && chmod +x /usr/bin/doxygen

RUN wget -O- http://release.fabu.ai/deps/tmux-2.8.tar.gz | tar -xazC /usr -f -

# vnc
RUN apt-get update && apt-get install --no-upgrade --no-install-recommends -y \
    lubuntu-core gnome-terminal && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -O http://release.fabu.ai/deps/turbovnc_2.1.2_amd64.deb \
        -O http://release.fabu.ai/deps/libjpeg-turbo-official_1.5.2_amd64.deb \
        -O http://release.fabu.ai/deps/virtualgl_2.5.2_amd64.deb \
        -O http://release.fabu.ai/deps/virtualgl32_2.5.2_amd64.deb && \
    dpkg -i *.deb && \
    rm -f /tmp/*.deb && \
    sed -i 's/$host:/unix:/g' /opt/TurboVNC/bin/vncserver

ENV PATH ${PATH}:/opt/VirtualGL/bin:/opt/TurboVNC/bin

RUN wget -O /etc/X11/org.conf http://release.fabu.ai/deps/vnc/xorg.conf && \
    wget -O /usr/share/lubuntu/wallpapers/1604-lubuntu-default-wallpaper.png http://release.fabu.ai/deps/vnc/background.png

RUN wget -O- http://release.fabu.ai/deps/osqp.tar.gz | tar -zxvf - -C /usr/local/ && echo '/usr/local/osqp/lib' >> /etc/ld.so.conf.d/osqp.conf && ldconfig

RUN mkdir /usr/local/code-server && wget -O- http://release.fabu.ai/deps/code-server.tar.gz | tar -zxvf - --strip-components=1 -C /usr/local/code-server

RUN echo /opt/roadstar-platform/ros/lib > /etc/ld.so.conf.d/ros.conf && ldconfig 

RUN echo 'X11UseLocalhost no' >> /etc/ssh/sshd_config

ENTRYPOINT ["/roadstar/docker/scripts/entrypoint.sh"]

CMD ["/bin/bash"]
