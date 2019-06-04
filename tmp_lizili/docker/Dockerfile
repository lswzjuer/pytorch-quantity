FROM nvcr.io/nvidia/pytorch:19.03-py3

RUN rm -rf /var/lib/apt/lists/* \
    /etc/apt/sources.list.d/cuda.list \
    /etc/apt/sources.list.d/nvidia-ml.list && \
    sed -i s@/archive.ubuntu.com/@/mirrors.tuna.tsinghua.edu.cn/@g /etc/apt/sources.list && \
    apt-get update && \

# -----------------------
# tool
# -----------------------
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         zsh \
         sudo \
         ctags \
         htop \
         tmux \
         ssh \
         xterm \
         zip \
         unzip \
         wget \
         gdb \
         bc \
         man \
         less \
         debconf-utils \
         locate \
         silversearcher-ag \
         ca-certificates \
         libboost-all-dev \
         libjpeg-dev \
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
         lsof \
         libxext-dev \
         libx11-dev \
         xauth \
         x11-utils \
         x11proto-gl-dev \
         libpng-dev \
         php7.0 \
         php7.0-curl \
         libsparsehash-dev \
         gosu

# arc
RUN mkdir /usr/local/arc && cd /usr/local/arc && \
    git clone http://git.fabu.ai:7070/third_party/arcanist.git && \
    git clone https://github.com/phacility/libphutil.git && \
    ln -s /usr/local/arc/arcanist/bin/arc /usr/bin/arc

# -------------------------------
# python
# -------------------------------
RUN  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common \
        && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.6 \
        python3.6-dev \
        python3.6-tk \
        && \
    wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/get-pip.py && \
    python3.6 ~/get-pip.py && \
    ln -s /usr/bin/python3.6 /usr/local/bin/python3 

# --------------------------------------
# python lib
# --------------------------------------
RUN echo "[global] timeout = 6000 \
index-url = https://pypi.tuna.tsinghua.edu.cn/simple \
" > /etc/pip.conf && \
    python3 -m pip --no-cache-dir install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade \
        setuptools \
        && \
    python3 -m pip --no-cache-dir install -i https://pypi.tuna.tsinghua.edu.cn/simple \
        numpy \
        numba \
        scipy \
        pandas \
        cloudpickle \
        scikit-learn==0.20.1 \
        matplotlib \
        Cython \
        shapely \
        plotly \
        loguru \
        opencv-python \
        tqdm \
        easydict \
		glob2 \
        future \
        protobuf \
        enum34 \
        typing \
        torchvision \
		ipython \
		jupyter \
		fire \
        tensorboardX \
        ninja \
		yacs \
        jupyterlab \
        tensorflow \
        scikit-image \
        pybind11 \
        spyder-kernels \
        mxnet

RUN git clone --depth=1 https://github.com/robbyrussell/oh-my-zsh.git 
RUN mkdir /usr/local/code-server && wget -O- http://release.fabu.ai/deps/code-server.tar.gz | tar -zxvf - --strip-components=1 -C /usr/local/code-server

# --------------------------
# coco
# --------------------------------------
RUN git clone https://github.com/cocodataset/cocoapi.git &&\
    cd cocoapi/PythonAPI &&\
    python3 setup.py build_ext install &&\
    cd ../.. && rm -rf cocoapi

RUN pip install --no-cache-dir git+https://github.com/nvidia/retinanet-examples

# ------------------------------
# config & cleanup
# --------------------------------------
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple &&\
    ldconfig && \
	updatedb && \
    apt-get clean && \
    apt-get autoremove && \
	sed -i 's/Port 22/Port 2222/g' /etc/ssh/sshd_config && \
	rm -rf /var/lib/apt/lists/* /tmp/* ~/*

