#! /bin/bash
if [ $# != 1 ] 
then 
	echo "get $# params"
	echo "Cgroup Config Usage: $0 config_file"
	exit 1
fi

mount_dir="/sys/fs/cgroup/"
subsystem="cpuset/"
file_name="cgroup.procs"
prog_name=$0
infile=$1
default_mem=`cat "$mount_dir${subsystem}cpuset.mems"`
echo "default meminfo: $default_mem"

while read line
do
	data=${line%%\#*}  # remove comment begin with '#'
	data=`echo $data | tr -d '[ \t]'`  # remove space and tab
	if [[ -z $data ]]; then  # empty line
		continue
	fi
	module_name=${data%%:*}  # chars at the left of ':'
	value=${data#*:}         # chars at the the right of ':'

	# check whether module's cpuset exists
	dir=$mount_dir$subsystem$module_name
	if [ ! -d $dir ] ; then  # path not exists
		echo "cgroup $module_name not exists, create now!"
		mkdir "$dir"
	fi

	# set value for this control group
	echo "$value" > "$dir/cpuset.cpus"
	echo "$default_mem" > "$dir/cpuset.mems"  # must be set beefore wirte pid
	echo "group $module_name set! With cpus: $value, mems: $default_mem"

	# find all moduls process and push them into their cgroup
	for pid in `pgrep $module_name`
	do
		echo "$pid" > "$dir/$file_name"
		echo "add process $pid into $module_name"
	done
done < $infile


