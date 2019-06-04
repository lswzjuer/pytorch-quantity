#!/usr/bin/env bash

SHELL_PATH="$( cd "$( dirname "$0"  )" && pwd  )"

DOCKER_HOME="/home/$USER"
if [ "$USER" == "root" ];then
    DOCKER_HOME="/root"
fi

if [ -z $DOCKER_NAME ];then
    DOCKER_NAME="${USER}_docker"
fi

DOCKER_REPO="docker.fabu.ai:5000/roadtensor/roadtensor"
VERSION="vision-20190516_2218"
IMG=${DOCKER_REPO}:$VERSION
LOCAL_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
DOCKER_WORKDIR="/roadtensor"

USER_ID=$(id -u)
GRP=$(id -g -n)
GRP_ID=$(id -g)

function local_volumes() {
  volumes="-v $LOCAL_DIR:$DOCKER_WORKDIR \
           -v $HOME/.ssh:${DOCKER_HOME}/.ssh \
		   -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
		   -v /media:/media \
		   -v /etc/localtime:/etc/localtime:ro \
		   -v /private:/private \
		   -v /onboard_data:/onboard_data \
		   -v /nfs:/nfs \
		   -v ${HOME}/.torch:${DOCKER_HOME}/.torch \
		   -v /data:/data"

  echo "${volumes}"
}

function add_user() {
  add_script="addgroup --gid ${GRP_ID} ${GRP} && \
      adduser --disabled-password --gecos '' ${USER} \
        --uid ${USER_ID} --gid ${GRP_ID} 2>/dev/null && \
      usermod -aG sudo ${USER} && \
      echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
      cp -r /etc/skel/. /home/${USER} && \
      chsh -s /usr/bin/zsh ${USER} && \
      chown -R ${USER}:${GRP} '/home/${USER}'"
  echo "${add_script}"
}

function config_zsh() {
  config_script="cp -r /workspace/oh-my-zsh .oh-my-zsh && \
      cp .oh-my-zsh/templates/zshrc.zsh-template .zshrc && \
      sed -i 's/\"robbyrussell/\"candy/g' .zshrc"
  echo "${config_script}"
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

    DOCKER_CMD="nvidia-docker"
    if ! [ -x "$(command -v ${DOCKER_CMD})" ]; then
      DOCKER_CMD="docker"
    fi

    # check whether nvidia driver version is higher than 418.
    if [ -f /sys/module/nvidia/version ]; then
      ok=$(echo "$(cat /sys/module/nvidia/version | cut -c 1-5) > 418.0" | bc)
      if [ "$ok" != "1" ]; then
        DOCKER_CMD="docker"
        info "Your nvidia driver version is below 418.0, please upgrade your nvidia driver to use nvidia docker."
      fi
    fi

    LOCAL_HOST=`hostname`
    eval ${DOCKER_CMD} run -it \
        -d \
        --name ${DOCKER_NAME}\
        -e DISPLAY=$display \
        -e DOCKER_USER=$USER \
        -e USER=$USER \
        -e DOCKER_USER_ID=$USER_ID \
        -e DOCKER_GRP=$GRP \
        -e DOCKER_GRP_ID=$GRP_ID \
        -e DOCKER_HOME=$DOCKER_HOME \
        -e SSH_AUTH_SOCK=/tmp/.ssh-agent-$USER/agent.sock \
        $(local_volumes) \
        -p :2222 \
        -p :6006 \
        -p :8443 \
        -w $DOCKER_WORKDIR \
        --dns=114.114.114.114 \
        --add-host in_docker:127.0.0.1 \
        --add-host ${LOCAL_HOST}:127.0.0.1 \
        --hostname in_docker \
        --ipc=host \
        $IMG

    docker exec ${DOCKER_NAME} service ssh start
    if [ "${USER}" != "root" ]; then
        docker exec ${DOCKER_NAME} bash -c "$(add_user)"
    fi

    docker exec -u ${USER} -w ${DOCKER_HOME} ${DOCKER_NAME} bash -c "$(config_zsh)"
    docker exec -d -u $USER ${DOCKER_NAME} /usr/local/code-server/code-server -HN /roadtensor
    docker exec -d -u $USER ${DOCKER_NAME} python setup.py build develop --user

    # add datasets and models dir
    docker exec ${DOCKER_NAME} bash -c "sudo mkdir -p /datasets && sudo chown ${USER}:${USER} /datasets"
	docker cp -L ~/.gitconfig ${DOCKER_NAME}:${DOCKER_HOME}/.gitconfig
	docker cp -L ~/.vimrc ${DOCKER_NAME}:${DOCKER_HOME}/.vimrc
	docker cp -L ~/.vim ${DOCKER_NAME}:${DOCKER_HOME}

	# port
	echo "*******************"
    docker port ${DOCKER_NAME} | sed "s/2222[^:]*/ssh/" | sed "s/6006[^:]*/tensorboard/" | sed "s/8443[^:]*/vscode/" | tee ${DOCKER_HOME}/.roadtensor_port
	echo "*******************"
	# so that we can also find out ports inside docker
	docker cp -L ~/.roadtensor_port ${DOCKER_NAME}:${DOCKER_HOME}/.roadtensor_port

}

main
