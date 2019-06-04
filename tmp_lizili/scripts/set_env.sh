export PYTHONPATH=/roadstar/bazel-genfiles:/roadstar/release/python/bazel-genfiles:/opt/roadstar-platform/ros/lib/python2.7/dist-packages:/usr/local/opencv-3.3.1/lib/python2.7/dist-packages

COMMON_LIB_PATH="/usr/local/lib:/opt/roadstar-platform/ros/lib:/roadstar/release/lib"
THIRD_PARTY_LIB_PATH="/usr/local/tensorflow/lib:/usr/local/tensorrt/lib:/usr/local/caffe/lib:/usr/local/mxnet/lib"
CUDA_LIB_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs"
if [ -e /usr/local/cuda/ ];then
  export PATH=/opt/roadstar-platform/ros/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  export LD_LIBRARY_PATH=$COMMON_LIB_PATH:$CUDA_LIB_PATH:$THIRD_PARTY_LIB_PATH
  export C_INCLUDE_PATH=/usr/local/cuda/include
  export CPLUS_INCLUDE_PATH=/usr/local/cuda/include
else
  export PATH=/opt/roadstar-platform/ros/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  export LD_LIBRARY_PATH=$COMMON_LIB_PATH
  export C_INCLUDE_PATH=""
  export CPLUS_INCLUDE_PATH=""
fi

# only ros domain id is the same, ros nodes can receive msgs each other
# export ROS_DOMAIN_ID=1000
ip=$(ifconfig | grep -Po '192.168.[35].\d*' | sort -V | head -n 1)
if [ ! -z "$ip" ]; then
  echo "using $ip as ROS ip"
  export ROS_IP=$ip
  if [ "$ip" = "192.168.3.3"  -o "$ip" = "192.168.3.4" ]; then
    echo "using 192.168.3.2 as master"
    export ROS_MASTER_URI="http://192.168.3.2:11311"
  elif [ "$ip" = "192.168.5.4" ]; then
    echo "using 192.168.5.2 as master"
    export ROS_MASTER_URI="http://192.168.5.2:11311"
  else
    echo "using $ip as master"
    export ROS_MASTER_URI="http://$ip:11311"
  fi
fi
