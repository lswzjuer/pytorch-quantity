#!/usr/bin/env bash



#! /bin/bash

python result2pb.py ../../control/conf/dongfeng.pb.txt $1

echo "Created control conf file: control_conf_pb.txt"
echo "with updated calibration table"
