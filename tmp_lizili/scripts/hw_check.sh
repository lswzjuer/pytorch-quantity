#!/usr/bin/env bash




DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "${DIR}/.."

source "$DIR/roadstar_base.sh"
LOG="${ROADSTAR_ROOT_DIR}/data/log/hw_check.out"

case $1 in
  "can")
    # setup can device
    if [ ! -e /dev/can0 ]; then
      sudo mknod --mode=a+rw /dev/can0 c 52 0
    fi

    eval "${ROADSTAR_BIN_PREFIX}/modules/monitor/hwmonitor/hw_check/can_check | tee ${LOG}"
    ;;
  "gps")
    eval "${ROADSTAR_BIN_PREFIX}/modules/monitor/hwmonitor/hw_check/gps_check | tee ${LOG}"
    ;;
  *)
    echo "Usage: $0 {can|gps}" | tee "${LOG}"
    exit 1
    ;;
esac
