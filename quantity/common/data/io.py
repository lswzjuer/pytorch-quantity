""" 
* Copyright 2019 The roadtensor Authors. All Rights Reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
"""

import sys
sys.path.insert(0, '/private/xuguodong/proto')
import os.path as osp

import modules.common.proto.sensor_source_pb2 as sensor_source_pb2
from modules.perception_v2.data.proto.transform_pb2 import Transform
from modules.perception_v2.data.proto.camera_conf_pb2 import CameraConf
from modules.perception_v2.data.proto.lidar_frame_label_pb2 import LidarFrameLabel, _LIDARFRAMELABEL
from modules.common.coordinate.proto.calibration_config_pb2 import CalibrationConfigs
from google.protobuf import text_format

import cv2
import numpy as np


def read_lidar_pointcloud(pointcloud_filename):
    """read lidar pointcloud.

    Args:
        pointcloud_filename (str): path of pointcloud bin file.

    Return: 
        raw_points (numpy.array): pointcloud numpy arrary 

    """
    with open(pointcloud_filename, 'rb') as fr:
        raw_points = np.fromfile(fr, np.float32).reshape(-1, 4)
    return raw_points


def read_3d_label(lidar_frame_label_filename,
                  type_list=['Pedestrian', 'Cyclist', 'Car', 'Truck']):
    """read 3d label

    Args:
        lidar_frame_label_filename (str): path of 3d label protobuf file.
        type_list (List): list of obstacle type needed.

    Return: 
        obstacle_list (List): list of obstacle object in type_list 

    """
    lidar_frame_label = LidarFrameLabel()
    with open(lidar_frame_label_filename, "r") as fr:
        text_format.Parse(fr.read(), lidar_frame_label)

    obstacle_type_map = _LIDARFRAMELABEL.enum_types_by_name["ObstacleType"]

    obstacle_list = []
    for obstacle in lidar_frame_label.obstacles:
        obstacle.type_str = obstacle_type_map.values_by_number[obstacle.type].name
        if obstacle.type_str not in type_list:
            continue
        obstacle_list.append(obstacle)

    return obstacle_list

def convert_obstacle_to_gt_boxes(obstacle_list):
    return np.array(
        [[obj.x, obj.y, obj.z, obj.width, obj.length, obj.height, obj.rotation]
         for obj in obstacle_list]
    )

# TODO: more test, more mode, Gray, RGB for example
# 77 ms ± 541 µs per loop for cv2.imread(img_path, 1)[..., [2, 1, 0]]
# 85.3 ms ± 1.18 ms per loop for Image.open(img_path).convert("RGB")
def read_img(img_filename, gray=False, rgb=False):
    """Read image

    """
    if gray:
        return cv2.imread(img_filename, 0)
    img = cv2.imread(img_filename, 1)
    if rgb:
        img = img[..., [2, 1, 0]]
    return img


def read_calib_config(calibration_dirname, sensor=False, camera=True):
    """read calib config

    Args:
        calibration_conf_dirname (str): path of calibration base dir.
        sensor (bool): whether return sensor calibration filename.
        camera (bool): whether return camera calibration filename.

    Return: 
        calib_config_dict (Dict): key is sensor name, value is path of calibration file name

    """
    calibration_conf_filename = osp.join(
        calibration_dirname, "calibration_conf.config")
    calib_config = CalibrationConfigs()
    with open(calibration_conf_filename, 'r') as fr:
        text_format.Parse(fr.read(), calib_config)

    calib_config_dict = {}
    type_map = sensor_source_pb2.DESCRIPTOR.enum_types_by_name["SensorSource"]

    if sensor:
        for sensor_file in calib_config.sensor_file_config.sensor_files:
            sensor_type_name = type_map.values_by_number[sensor_file.sensor_name].name
            calib_config_dict[sensor_type_name] = osp.join(
                calibration_dirname, sensor_file.file_name)

    if camera:
        for camera_file in calib_config.sensor_file_config.camera_files:
            camera_type_name = type_map.values_by_number[camera_file.sensor_name].name
            calib_config_dict[camera_type_name] = osp.join(
                calibration_dirname, camera_file.file_name)

    return calib_config_dict


def read_camera_calib(camera_calib_filename):
    """read camera calib file

    Args:
        camera_calib_filename (str): path of camera calib filename.

    Return: 
        camera_conf

    """
    camera_conf = CameraConf()
    with open(camera_calib_filename, 'r') as fr:
        text_format.Parse(fr.read(), camera_conf)
    return camera_conf


def read_lidar_calib(lidar_imu_calib_filename):
    """read lidar to imu calib file

    Args:
        lidar_imu_calib_filename (str): path of lidar to imu calib filename.

    Return: 
        lidar2imu

    """
    lidar_calib = Transform()
    with open(lidar_imu_calib_filename, 'r') as fr:
        text_format.Parse(fr.read(), lidar_calib)
    return lidar_calib
