basepath=$(cd `dirname $0`; pwd)

bash $basepath/control.sh stop
sleep 1s

bash $basepath/planning.sh stop
sleep 1s

bash $basepath/perception.sh stop
sleep 1s

bash $basepath/hdmap.sh stop
sleep 1s

bash $basepath/record_bag.sh stop
