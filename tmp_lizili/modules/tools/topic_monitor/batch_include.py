#!/usr/bin/env python

from std_msgs.msg import String
from camera_msgs.msg import ImageFrame
from camera_msgs.msg import CompressedImageFrame
from velodyne_msgs.msg import PointCloud
from velodyne_msgs.msg import VelodyneScanUnified
from pandar_msgs.msg import PandarScan
from pandar_msgs.msg import Pandar40pScan
from modules.msgs.drivers.proto import conti_radar_pb2
from modules.msgs.drivers.proto import delphi_esr_pb2
from modules.msgs.drivers.lidar.proto import lidar_scan_pb2
from modules.msgs.drivers.lidar.proto import point_cloud_pb2
from modules.msgs.canbus.proto import chassis_pb2
from modules.msgs.drivers.novatel.proto import gnss_best_pose_pb2
from modules.msgs.drivers.novatel.proto import gnss_status_pb2
from modules.msgs.localization.proto import localization_pb2
from modules.msgs.perception.proto import fusion_map_pb2


topic_infos = [
    {
        'module_name': 'CAMERA_HL',
        'topic_name': '/roadstar/drivers/pylon_camera/camera/frame/head_left/jpg',
        'proto': CompressedImageFrame,
        'hz_mean': 13,
        'hz_dev': 2,
    },
    {
        'module_name': 'CAMERA_HR',
        'topic_name': '/roadstar/drivers/pylon_camera/camera/frame/head_right/jpg',
        'proto': CompressedImageFrame,
        'hz_mean': 10,
        'hz_dev': 2,
    },
    {
        'module_name': 'CAMERA_FR',
        'topic_name': '/roadstar/drivers/pylon_camera/camera/frame/front_right/jpg',
        'proto': CompressedImageFrame,
        'hz_mean': 10,
        'hz_dev': 2,
    },
    {
        'module_name': 'CAMERA_FL',
        'topic_name': '/roadstar/drivers/pylon_camera/camera/frame/front_left/jpg',
        'proto': CompressedImageFrame,
        'hz_mean': 10,
        'hz_dev': 2,
    },
    {
        'module_name': 'CAMERA_TL',
        'topic_name': '/roadstar/drivers/pylon_camera/camera/frame/tail_left/jpg',
        'proto': CompressedImageFrame,
        'hz_mean': 10,
        'hz_dev': 2,
    },
    {
        'module_name': 'CAMERA_TR',
        'topic_name': '/roadstar/drivers/pylon_camera/camera/frame/tail_right/jpg',
        'proto': CompressedImageFrame,
        'hz_mean': 10,
        'hz_dev': 2,
    },
    {
        'module_name': 'LIDAR_MAIN',
        'topic_name': '/roadstar/drivers/lidar/pointcloud/main',
        'proto': PointCloud,
        'hz_mean': 10,
        'hz_dev': 2,
    },
    {
        'module_name': 'LIDAR_TL',
        'topic_name': '/roadstar/drivers/lidar/pointcloud/tail_left',
        'proto': PointCloud,
        'hz_mean': 10,
        'hz_dev': 2,
    },
    {
        'module_name': 'LIDAR_TR',
        'topic_name': '/roadstar/drivers/lidar/pointcloud/tail_right',
        'proto': PointCloud,
        'hz_mean': 10,
        'hz_dev': 2,
    },
    {
        'module_name': 'LIDAR_HM',
        'topic_name': '/roadstar/drivers/lidar/pointcloud/head_mid',
        'proto': PointCloud,
        'hz_mean': 10,
        'hz_dev': 2,
    },
    {
        'module_name': 'RADAR_TM',
        'topic_name': '/roadstar/drivers/conti_radar',
        'proto': conti_radar_pb2.ContiRadar,
        'hz_mean': 13.7,
        'hz_dev': 3,
    },
    {
        'module_name': 'RADAR_HM',
        'topic_name': '/roadstar/drivers/conti_radar/head_mid',
        'proto': conti_radar_pb2.ContiRadar,
        'hz_mean': 13.7,
        'hz_dev': 3,
    },
    {
        'module_name': 'CANBUS',
        'topic_name': '/roadstar/canbus/chassis',
        'proto': chassis_pb2.Chassis,
        'hz_mean': 100,
        'hz_dev': 20,
    },
    {
        'module_name': 'NOV_RAW',
        'topic_name': '/roadstar/drivers/novatel/raw_data',
        'proto': String,
        'hz_mean': 330,
        'hz_dev': 66,
    },
    {
        'module_name': 'NOV_BEST',
        'topic_name': '/roadstar/driver/novatel/best_pose',
        'proto': gnss_best_pose_pb2.GnssBestPose,
        'hz_mean': 10,
        'hz_dev': 2,
    },
    {
        'module_name': 'NOV_STA',
        'topic_name': '/roadstar/driver/novatel/ins_status',
        'proto': gnss_status_pb2.InsStatus,
        'hz_mean': 10,
        'hz_dev': 2,
    },
    {
        'module_name': 'LOC',
        'topic_name': '/roadstar/localization',
        'proto': localization_pb2.Localization,
        'hz_mean': 200,
        'hz_dev': 40,
    },
    {
        'module_name': 'PERCEP',
        'topic_name': '/roadstar/perception/fusion_map',
        'proto': fusion_map_pb2.FusionMap,
        'hz_mean': 10,
        'hz_dev': 2,
    }
]
