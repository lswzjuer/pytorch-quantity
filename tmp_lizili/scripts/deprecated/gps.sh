#!/usr/bin/env bash




DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "${DIR}/.."

source "${DIR}/roadstar_base.sh"

function start() {
    LOG="${ROADSTAR_ROOT_DIR}/data/log/gnss_driver.out"
    CMD="roslaunch gnss_driver gnss.launch"
    NUM_PROCESSES="$(pgrep -c -f "gnss_nodelet_manager")"
    if [ "${NUM_PROCESSES}" -eq 0 ]; then
       eval "nohup ${CMD} </dev/null >${LOG} 2>&1 &"
    fi
}

function stop() {
    pkill -f gnss_driver
}

# run command_name module_name
function run() {
    case $1 in
        start)
            start
            ;;
        stop)
            stop
            ;;
        *)
            start
            ;;
    esac
}

run "$1"
