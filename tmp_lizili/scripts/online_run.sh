basepath=$(cd `dirname $0`; pwd)

$basepath/hdmap.sh
sleep 1s

$basepath/perception.sh
sleep 1s

$basepath/control.sh
sleep 1s

$basepath/planning.sh
sleep 1s

$basepath/record_bag.sh

