#!/bin/bash
# SAMPLE RUN: ./run_coverage.sh //modules/perception_v2/... modules/perception_v2/common

if [ $# -ne "2" ]; then
  echo $0 TARGET FILTER
  echo "SAMPLE RUN: ./run_coverage.sh //modules/perception_v2/... modules/perception_v2/common"
  exit 1
fi

TARGETS=$1
PATTERN=$2
WORKDIR=/tmp/coverage
ESCAPED_WORKDIR=$(echo /tmp/coverage | sed 's/\//\\\//g')

rm -rf $WORKDIR
mkdir $WORKDIR

cd /roadstar

find bazel-out/ | grep gcno | xargs rm 2>/dev/null

bazel build --collect_code_coverage $TARGETS || exit 1

find bazel-out/ | grep gcno | grep -v external > $WORKDIR/gcno.list
rsync -az --files-from=$WORKDIR/gcno.list /roadstar/ $WORKDIR

# generating empty baseline
find $WORKDIR/bazel-out/ | grep -v $PATTERN | xargs rm 2>/dev/null
/usr/bin/lcov -q -c -i --no-external -d $WORKDIR -o $WORKDIR/baseline.info.1 2>/dev/null
/usr/bin/lcov -q -e $WORKDIR/baseline.info.1 "*_objs*$PATTERN*$PATTERN*" -o $WORKDIR/baseline.info.2
/usr/bin/lcov -q -r $WORKDIR/baseline.info.2 "*external*" "*.pb.h" "*.pb.cc" "*_test.cc" -o $WORKDIR/baseline.info  
sed -i "s/\/.*\/modules\/.*\/modules\//\/roadstar\/modules\//g" $WORKDIR/baseline.info

for TARGET in $(bazel query "kind(cc_.*, tests($TARGETS))"); do 
  echo "Running target: " $TARGET
  bin=$(echo $TARGET | sed 's/\/\//bazel-bin\//g' | sed 's/:/\//g')
  GCOV_PREFIX_STRIP=3 GCOV_PREFIX=$WORKDIR $bin
  if [ "$?" -ne "0" ]; then
    echo "Failed target: " $TARGET
    exit 1;
  fi

  name=$(echo $bin | sed 's/\//_/g')
 
  find $WORKDIR/bazel-out/ | grep -v $PATTERN | xargs rm 2>/dev/null
  /usr/bin/lcov -q -c --no-external -d $WORKDIR -o $WORKDIR/"$name".info.1 2>/dev/null
  /usr/bin/lcov -q -e $WORKDIR/"$name".info.1 "*_objs*$PATTERN*$PATTERN*" -o $WORKDIR/"$name".info.2
  /usr/bin/lcov -q -r $WORKDIR/"$name".info.2 "*external*" "*.pb.h" "*.pb.cc" "*_test.cc" -o $WORKDIR/"$name".info  
  sed -i "s/\/.*\/modules\/.*\/modules\//\/roadstar\/modules\//g" $WORKDIR/"$name".info
  find $WORKDIR | grep .gcda | xargs rm 2>/dev/null
done


MERGE_CMD=lcov
for i in $(ls $WORKDIR/*.info); do
  MERGE_CMD="$MERGE_CMD --add-tracefile $i"
done
MERGE_CMD="$MERGE_CMD -o $WORKDIR/all.info"
$MERGE_CMD

OUTPUT_HTML=$(mktemp -d)
genhtml $WORKDIR/all.info --output-directory=$OUTPUT_HTML --prefix=/roadstar --config-file=$(dirname $0)/genhtml.conf

echo HTMLs at $OUTPUT_HTML
