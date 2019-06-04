#!/bin/bash

WORKDIR=/tmp

function install() {
  cd $WORKDIR
  wget http://release.fabu.ai/deps/linuxcan.tar.gz

  tar -zxvf linuxcan.tar.gz
  cd linuxcan
  sudo make uninstall
  make clean
  make
  sudo make install
  sudo make load
  cd $WORKDIR
	rm -rf linuxcan.tar.gz linuxcan
  echo "Done!"
}

function print_usage() {
  RED='\033[0;31m'
  BLUE='\033[0;34m'
  BOLD='\033[1m'
  NONE='\033[0m'

  echo -e "\n${RED}Usage${NONE}:
  .${BOLD}/install_kvaser_driver.sh${NONE} [OPTION]"

  echo -e "\n${RED}Options${NONE}:
  ${BLUE}install${NONE}: install kvaser driver
  "
}


function run() {
    case $1 in
        install)
           install 
            ;;
        print_usage)
            print_usage
            ;;
        *)
            install 
            ;;
    esac
}

run "$1"
