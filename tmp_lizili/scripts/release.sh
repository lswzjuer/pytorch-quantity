#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${DIR}/.."

RELEASE_DIR="${HOME}/.cache/local_roadstar_release"
if [ -d "${RELEASE_DIR}" ]; then
  rm -fr "${RELEASE_DIR}"
fi
ROADSTAR_DIR="${RELEASE_DIR}/roadstar"
mkdir -p "${ROADSTAR_DIR}"

# Find binaries and convert from //path:target to path/target
BINARIES=$(bazel query "kind(cc_binary, //...)" | sed 's/^\/\///' | sed 's/:/\//')
# Copy binaries to release dir.
for BIN in ${BINARIES}; do
  SRC_PATH="bazel-bin/${BIN}"
  DST_PATH="${ROADSTAR_DIR}/${BIN}"
  if [ -e "${SRC_PATH}" ]; then
    mkdir -p "$(dirname "${DST_PATH}")"
    cp "${SRC_PATH}" "${DST_PATH}"
  fi
done

# modules data and conf
MODULES_DIR="${ROADSTAR_DIR}/modules"
mkdir -p $MODULES_DIR
for m in common control localization perception_v2 dreamview \
     planning monitor hdmap vehicle calibration map_v2 
do
  TARGET_DIR=$MODULES_DIR/$m
  mkdir -p $TARGET_DIR
  if [ -d modules/$m/conf ]; then
    cp -r modules/$m/conf $TARGET_DIR
  fi
  if [ -d modules/$m/data ]; then
    cp -r modules/$m/data $TARGET_DIR
  fi
  if [ -e modules/$m/README.md ]; then
    cp modules/$m/README.md $TARGET_DIR
  fi
done

# remove all pyc file in modules/
find modules/ -name "*.pyc" | xargs -I {} rm {}

# tools
cp -r modules/tools $MODULES_DIR

# scripts
cp -r scripts ${ROADSTAR_DIR}

cp -Lr bazel-bin/modules/dreamview/dreamview.runfiles/roadstar/modules/dreamview/frontend $MODULES_DIR/dreamview

# python
PYTHON_DIR="${ROADSTAR_DIR}/python"
mkdir "${PYTHON_DIR}"
for f in $(find bazel-genfiles/modules | grep -E '(_pb2.py|_pb2_grpc.py|__init__.py|lib_message_pyutils)'); do
  mkdir -p $PYTHON_DIR/$(dirname $f);
  cp $f $PYTHON_DIR/$(dirname $f);
done
find $PYTHON_DIR/bazel-genfiles/* -type d -exec touch "{}/__init__.py" \;


# lib
LIB_DIR="${ROADSTAR_DIR}/lib"
mkdir "${LIB_DIR}"

cp -r bazel-genfiles/external $LIB_DIR
cp bazel-bin/modules/perception_v2/cuda_util/libintegrated_cuda_util.so $LIB_DIR
cp third_party/ACKNOWLEDGEMENT.txt "${ROADSTAR_DIR}"

while read line
do 
  if [ "$line" = "" ]; then
    echo "Null Line"
  else
    mkdir -p $MODULES_DIR/${line}
    rm -rf $MODULES_DIR/${line}
    echo "cp -r modules/${line} $MODULES_DIR/${line}"
    cp -r modules/${line} $MODULES_DIR/${line}
  fi
done < $DIR/release_path

# release info
META="${ROADSTAR_DIR}/meta.ini"
echo "git_commit: $(git rev-parse HEAD)" >> $META
# resource info
echo "resource_commit: $(cd resources; git rev-parse HEAD)" >> $META
# release time
echo "release_generation_time: $(date)" >> $META
