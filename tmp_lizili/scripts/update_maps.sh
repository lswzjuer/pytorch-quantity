#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RELEASE_PATH=~/.cache/roadstar_release/maps
RELEASE_REMOTE_PATH='http://map.fabu.ai:8899'
VERSION_FILE='maps_version'

# update roadstar release
mkdir -p $RELEASE_PATH

# function to read version file
read_version(){
  if [ -f $1 ]; then
    while read map; do
      map=(${map})
      eval $2[${map[0]}]=${map[1]}
    done < $1
  fi
}

# maps version array
declare -A maps=()
declare -A old_maps=()
# backup old version and read it
if [ -f ${RELEASE_PATH}/${VERSION_FILE} ]; then
  # read files
  read_version ${RELEASE_PATH}/${VERSION_FILE} "old_maps"
fi

# fetch newest version
wget ${RELEASE_REMOTE_PATH}/${VERSION_FILE} -O ${RELEASE_PATH}/${VERSION_FILE}
read_version ${RELEASE_PATH}/${VERSION_FILE} "maps"
for map in "${!maps[@]}"; do
  if [[ ! -f ${RELEASE_PATH}/${map}.xml || ${maps[${map}]} > ${old_maps[${map}]} ]]; then
    echo "downloading ${map}"
    wget ${RELEASE_REMOTE_PATH}/${map} -O ${RELEASE_PATH}/${map}.xml
  fi
  unset old_maps["${map}"]
done

# remove useless maps
for old_map in "${!old_maps[@]}"; do
  echo "removing ${old_map}"
  rm -f ${RELEASE_PATH}/${old_map}.xml
done
