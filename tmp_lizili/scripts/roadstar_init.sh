#!/usr/bin/env bash




DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "${DIR}/.."

source "${DIR}/roadstar_base.sh"

USR_NAME=$(id -u -n)
GRP_NAME=$(id -g -n)

# grant caros user to access GPS device
if [ -e /dev/ttyUSB0 ]; then
    sudo chown ${USR_NAME}:${GRP_NAME} /dev/ttyUSB0 /dev/ttyUSB1
fi

# setup can device
if [ ! -e /dev/can0 ]; then
    sudo mknod --mode=a+rw /dev/can0 c 52 0
fi

if [ -e /dev/can0 ]; then
    sudo chown ${USR_NAME}:${GRP_NAME} /dev/can0
fi

# enable coredump
echo "${ROADSTAR_ROOT_DIR}/data/core/core_%e.%p" | sudo tee /proc/sys/kernel/core_pattern >/dev/null 2>&1
