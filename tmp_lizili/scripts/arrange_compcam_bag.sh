for x in $@
do
  echo $x 
  python ./modules/drivers/pylon_camera/camera_driver/camera_compression/scripts/rearrange_bag.py $x
done
