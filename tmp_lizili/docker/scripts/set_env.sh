# docker
export DOCKER_REPO="docker.fabu.ai:5000/roadstar/roadstar"
export VERSION_X86_64="dev-x86_64-20190531_2039"
export VERSION_AARCH64="dev-aarch64-20170712_1533"
export DOCKER_IMG_PATH="/tmp/roadstar_docker.img"

display_num(){
  DISPLAY_NUM=$(echo $DISPLAY | grep -oP '(?<=:)\d+' | head -n1)
  if [ -z $DISPLAY ] || ! [ -S /tmp/.X11-unix/X${DISPLAY_NUM} ]; then
    DISPLAY_NUM=$(cd /tmp/.X11-unix && ls X* 1>/dev/null 2>/dev/null && for x in X*; do echo "${x#X}"; done | awk 'max == "" || $1>max{max=$1} END{print max}')
  fi
  echo $DISPLAY_NUM
}

docker_display(){
  if [ "$DISPLAY" = ":$DISPLAY_NUM" ]; then
    echo "GUI will be displayed on local machine" >&2
    echo ":0"
  else
    echo "GUI will be displayed on dreamview" >&2
    echo ":1"
  fi
}

export DISPLAY_NUM="$(display_num)"
export DOCKER_DISPLAY="$(docker_display)"
export DISPLAY=":$DISPLAY_NUM"
