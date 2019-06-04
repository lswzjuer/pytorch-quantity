#!/usr/bin/env bash

SHELL_PATH="$( cd "$( dirname "$0"  )" && pwd  )"
source $SHELL_PATH/set_env.sh

VERSION=""
ARCH=$(uname -m)
DOCKER_HOME="/home/$USER"
if [ "$USER" == "root" ];then
    DOCKER_HOME="/root"
fi

if [ -z $DOCKER_NAME ];then
    DOCKER_NAME="${USER}_roadstar_dev"
fi


DATE=$(date +%F)

if [ ${ARCH} == "x86_64" ]; then
    VERSION=${VERSION_X86_64}
elif [ ${ARCH} == "aarch64" ]; then
    VERSION=${VERSION_AARCH64}
else
    echo "Unknown architecture: ${ARCH}"
    exit 0
fi

if [ -z "${DOCKER_REPO}" ]; then
    DOCKER_REPO=roadstar/roadstar
fi

IMG=${DOCKER_REPO}:$VERSION
LOCAL_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"

DATA_DIR=${LOCAL_DIR}/data
if [[ $# == 1 ]];then
  if [ -e "$1" ]; then
      DATA_DIR=$1
  else
      echo "Data path is nonexistent !!!"
  fi
  shift
fi

if [ ! -e "${DATA_DIR}/log/${DATE}" ]; then
    mkdir -p "${DATA_DIR}/log/${DATE}"
fi
if [ ! -e "${DATA_DIR}/bag" ]; then
    mkdir -p "${DATA_DIR}/bag"
fi
if [ ! -e "${DATA_DIR}/core" ]; then
    mkdir -p "${DATA_DIR}/core"
fi

source ${LOCAL_DIR}/scripts/roadstar_base.sh

function find_device() {
    # ${1} = device pattern
    local device_list=$(find /dev -name "${1}")
    if [ -z "${device_list}" ]; then
        warning "Failed to find device with pattern \"${1}\" ..."
    else
        local devices=""
        for device in $(find /dev -name "${1}"); do
            ok "Found device: ${device}."
            devices="${devices} --device ${device}:${device}"
        done
        echo "${devices}"
    fi
}

function local_volumes() {
  volumes="-v $LOCAL_DIR:/roadstar\
           -v $HOME/.cache:${DOCKER_HOME}/.cache\
           -v $HOME/.ssh:${DOCKER_HOME}/.ssh\
           -v /dev/null:/dev/raw1394\
           -v /tmp/core:/tmp/core\
           -v /tmp/.ssh-agent-$USER:/tmp/.ssh-agent-$USER"

  case "$(uname -s)" in
    Linux)
      volumes="${volumes} -v /tmp/.X11-unix/X${DISPLAY_NUM}:/tmp/.X11-unix/X0:rw \
                          -v /media:/media \
                          -v /run/udev:/run/udev:ro \
                          -v /etc/localtime:/etc/localtime:ro \
                          -v /private:/private \
                          -v /data:/data \
                          -v /onboard_data:/onboard_data"
      ;;
    Darwin)
      chmod -R a+wr ~/.cache/bazel
      ;;
  esac
  mkdir -p /tmp/.ssh-agent-$USER 2>&1 > /dev/null
  echo "${volumes}"
}

function get_port(){
  read lower_port upper_port < /proc/sys/net/ipv4/ip_local_port_range

  range=$1

  local success=false

  for (( port = lower_port ; port <= upper_port ; port+=$range )); do
    success=true
    for (( i = 0; i < range; i++ )); do
      netstat -ntl | grep $(expr $port + $i) 2>/dev/null >/dev/null && success=false && break
    done
    if [ $success = true ]; then
      for (( i = 0; i < range; i++ )); do
        echo $(expr $port + $i)
      done
      break
    fi
  done
}

function map_ports() {
  local ports=("$@")
  local port_num=${#ports[*]}
  local success=true
  for (( i = 0; i < ${port_num}; i++ )); do
    netstat -ntl | grep ${ports[${i}]} 2>/dev/null >/dev/null && success=false && break
  done
  if [ $success = true ]; then
    local avaliable_ports=("$@")
  else
    warning "Default port are used, using random ports instead." >&2
    local avaliable_ports=($(get_port ${port_num}))
  fi
  for (( i = 0; i < ${port_num}; i++ )); do
    echo "-p ${avaliable_ports[${i}]}:${ports[${i}]} "
  done
}

function main(){
    docker pull $IMG

    docker ps -a --format "{{.Names}}" | grep "${DOCKER_NAME}" 1>/dev/null
    if [ $? == 0 ]; then
        docker stop ${DOCKER_NAME} 1>/dev/null
        docker rm -f ${DOCKER_NAME} 1>/dev/null
    fi
    local display=""
    if [[ -z ${DISPLAY} ]];then
        display=":0"
    else
        display="${DISPLAY}"
    fi

    # enable coredump
    if [ ! -e "/tmp/core" ]; then
        mkdir -p /tmp/core
    fi
    CORE_DUMP_PATTER="/tmp/core/core_%e.%u.%s.%p"
    CUR_CORE_DUMP_PATTER=`cat /proc/sys/kernel/core_pattern`
    echo $CUR_CORE_DUMP_PATTER
    if [ ! "$CUR_CORE_DUMP_PATTER" == "$CORE_DUMP_PATTER" ]; then
        echo $CORE_DUMP_PATTER | sudo tee /proc/sys/kernel/core_pattern
    fi

    local devices=""
    devices="${devices} $(find_device 'ttyUSB*')"
    devices="${devices} $(find_device 'ttyS*')"
    devices="${devices} $(find_device 'can*')"
    devices="${devices} $(find_device '*pciefd*')"
    devices="${devices} $(find_device '*leaf*')"
    devices="${devices} $(find_device '*mhydra*')"
    devices="${devices} $(find_device 'ram*')"
    devices="${devices} $(find_device 'loop*')"
    devices="${devices} $(find_device 'video*')"
    devices="${devices} $(find_device 'camera*')"
    USER_ID=$(id -u)
    GRP=$(id -g -n)
    GRP_ID=$(id -g)
    LOCAL_HOST=`hostname`
    if [ ! -d "$HOME/.cache" ];then
        mkdir "$HOME/.cache"
    fi

    DOCKER_CMD="nvidia-docker"
    if ! [ -x "$(command -v ${DOCKER_CMD})" ]; then
      DOCKER_CMD="docker"
    fi

    # check whether nvidia driver version is higher than 410.
    if [ -f /sys/module/nvidia/version ]; then
      ok=$(echo "$(cat /sys/module/nvidia/version) > 410.0" | bc)
      if [ "$ok" != "1" ]; then
        DOCKER_CMD="docker"
        info "Your nvidia driver version is below 410.0, please upgrade your nvidia driver to use nvidia docker."
      fi
    fi
    info "using $DOCKER_CMD"
 
    DOMAIN_ID=`date +%N | cut -c1-7`

    info ROS_DOMAIN_ID=$DOMAIN_ID
    eval ${DOCKER_CMD} create -it \
        --name ${DOCKER_NAME}\
        -e DISPLAY=$display \
        -e DOCKER_USER=$USER \
        -e USER=$USER \
        -e DOCKER_USER_ID=$USER_ID \
        -e DOCKER_GRP=$GRP \
        -e DOCKER_GRP_ID=$GRP_ID \
        -e DOCKER_HOME=$DOCKER_HOME \
        -e ROS_DOMAIN_ID=$DOMAIN_ID \
        -e VEHICLE_NAME=$LOCAL_HOST \
        -e SSH_AUTH_SOCK=/tmp/.ssh-agent-$USER/agent.sock \
        $(local_volumes) \
        --ulimit core=-1 \
        $(map_ports 2222 6060 8888 8443) \
        -w /roadstar \
        ${devices} \
        --dns=114.114.114.114 \
        --add-host in_dev_docker:127.0.0.1 \
        --add-host ${LOCAL_HOST}:127.0.0.1 \
        --hostname in_dev_docker \
        --shm-size 2G \
        --security-opt seccomp=unconfined \
        $IMG

    docker cp -L ~/.gitconfig ${DOCKER_NAME}:${DOCKER_HOME}/.gitconfig
    docker cp -L ~/.arcrc ${DOCKER_NAME}:${DOCKER_HOME}/.arcrc
    docker cp -L /roadstar_ip ${DOCKER_NAME}:/roadstar/config
    docker start ${DOCKER_NAME}
 }

main
