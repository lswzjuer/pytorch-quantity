#!/bin/bash

head_left="0000001001"
head_right="0000001002"
mid_left="03121D3E00090721"
mid_right="06190F1600090720"
tail_right="0619173B00090720"
videolist='ls /dev/video*'
for video in $videolist
do
  serial=$(echo $(eval "udevadm info --query=all --name=$video | grep ID_SERIAL_SHORT"))
  len=${#serial}
  half=${serial:19:$len}
  echo $video
  echo $half
  if [[ $half == $head_left ]]
  then
    if [ -e "/dev/camera_head_left" ]
    then
      sudo rm /dev/camera_head_left
    fi
   sudo ln -s $video /dev/camera_head_left
  elif [[ $half == $head_right ]]
  then
    if [ -e "/dev/camera_head_right" ]
    then
      sudo rm /dev/camera_head_right
    fi
   sudo ln -s $video /dev/camera_head_right
  elif [[ $half == $mid_left ]]
  then
    if [ -e "/dev/camera_mid_left" ]
    then
      sudo rm /dev/camera_mid_left
    fi
   sudo ln -s $video /dev/camera_mid_left
  elif [[ $half == $mid_right ]]
  then
    if [ -e "/dev/camera_mid_right" ]
    then
      sudo rm /dev/camera_mid_right
    fi
   sudo ln -s $video /dev/camera_mid_right
  elif [[ $half == $tail_right ]]
  then
    if [ -e "/dev/camera_tail_right" ]
    then
      sudo rm /dev/camera_tail_right
    fi
   sudo ln -s $video /dev/camera_tail_right
  fi
done

  
