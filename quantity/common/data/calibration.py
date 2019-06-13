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

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 20:08:27 2019

@author: magicwang
"""
import cv2
import numpy as np
from roadtensor.common.data.io import (read_lidar_pointcloud, read_calib_config,
                                       read_camera_calib, read_lidar_calib)


class Calibration():

    def __init__(self, camera_calib, lidar2imu, ego2imu):
        camera_matrix = camera_calib.camera_matrix
        self.camera_matrix = np.array([[camera_matrix.fx, 0, camera_matrix.cx],
                                       [0, camera_matrix.fy, camera_matrix.cy],
                                       [0, 0, 1]])

        dist_coeffs_pb = camera_calib.dist_coeffs
        self.dist_coeffs = np.array([dist_coeffs_pb.k1, dist_coeffs_pb.k2, dist_coeffs_pb.p1,
                                     dist_coeffs_pb.p2, dist_coeffs_pb.k3])

        lidar_trans_pb = camera_calib.lidar64_trans
        self.r_vec = np.array(lidar_trans_pb.rvec)
        self.t_vec = np.array(lidar_trans_pb.tvec).reshape(3, 1)

        self.lidar_to_ego, self.lidar_to_imu, self.ego_to_imu = \
            self.get_transforms(lidar2imu, ego2imu)
        self.ego_to_lidar = np.linalg.inv(self.lidar_to_ego)
        self.imu_to_lidar = np.linalg.inv(self.lidar_to_imu)
        self.imu_to_ego = np.linalg.inv(self.ego_to_imu)
        self.update()

    def update(self):  # ready to support data augment
        self.r_mat = cv2.Rodrigues(self.r_vec)[0]
        self.RT = np.hstack([self.r_mat, self.t_vec])
        # p is the transform matrix (3 x 4) from ego to pixel, the most common projection matrix
        self.ego_to_pixel = np.dot(self.camera_matrix, self.RT).dot(self.ego_to_lidar)
        self.p = self.ego_to_pixel

    def get_transforms(self, lidar2imu, ego2imu):
        li_rvec = np.array([lidar2imu.roll, lidar2imu.pitch, lidar2imu.heading])
        li_tvec = np.array([lidar2imu.x, lidar2imu.y, lidar2imu.z])
        lidar_to_imu = np.zeros((4, 4), dtype=np.float32)
        lidar_to_imu[0:3, 0:3] = cv2.Rodrigues(li_rvec)[0]
        lidar_to_imu[0:3, -1] = li_tvec
        lidar_to_imu[3, 3] = 1

        ei_rvec = np.array([ego2imu.roll, ego2imu.pitch, ego2imu.heading])
        ei_tvec = np.array([ego2imu.x, ego2imu.y, ego2imu.z])
        ego_to_imu = np.zeros((4, 4), dtype=np.float32)
        ego_to_imu[0:3, 0:3] = cv2.Rodrigues(ei_rvec)[0]
        ego_to_imu[0:3, -1] = ei_tvec
        ego_to_imu[3, 3] = 1

        lidar_to_ego = np.dot(np.linalg.inv(ego_to_imu), lidar_to_imu)

        return lidar_to_ego, lidar_to_imu, ego_to_imu


def get_calibration(calib_dir):
    calib_config = read_calib_config(calib_dir, sensor=True, camera=True)
    camera_calib = read_camera_calib(calib_config['CameraHeadRight'])
    lidar2imu = read_lidar_calib(calib_config['LidarMain'])
    ego2imu = read_lidar_calib(calib_config['EgoFront'])
    return Calibration(camera_calib, lidar2imu, ego2imu)


def transform_coordinate(points, p):
    """
    :param points: points(3 x N) in some 3D coordinates
    :param p: homogeneous transformation matrix (4 x 4)
    :return: points(3 x N) in some other 3D coordinates
    """
    padded_points = np.append(points, np.ones((1, points.shape[1])), axis=0)
    homo_transformed_points = np.dot(p, padded_points)
    transformed_points = homo_transformed_points[0:3]
    return transformed_points


def project(points, p):
    """
    :param points: 3D points(3 x N) in some 3d coordinate
    :param p: projection matrix (3 x 4)
    :return: 2D points(2 x N) in image pixel coordinate
    """
    padded_points = np.append(points, np.ones((1, points.shape[1])), axis=0)
    points_2d = np.dot(p, padded_points)
    points_2d[0, :] = points_2d[0, :] / points_2d[2, :]
    points_2d[1, :] = points_2d[1, :] / points_2d[2, :]
    points_2d = np.delete(points_2d, 2, 0)
    points_2d = points_2d.astype(np.int32)
    return points_2d


# def unproject(points, calibration=None, z=None):
#     """Unproject 2D points(2 x N) in image pixel coordinate to 3D points(3 x N) in lidar coordinate"""
#     assert calibration is not None
#     if z is not None:
#         assert len(z) == points.shape[1]
#         valid_mask = z > 0.
#         z = z[valid_mask]
#         points = points[:, valid_mask]
#         R = calibration.r_mat
#         T = np.zeros_like(calibration.t_vec)
#         camera_matrix = calibration.camera_matrix
#         padded_points = np.vstack((points, np.ones(points.shape[1])))
#         points_camera = np.linalg.inv(camera_matrix).dot(padded_points)
#
#         RT_inv = np.hstack((R.T, -T))
#         padded_points_camera = np.vstack((points_camera, np.ones(points_camera.shape[1])))
#         tmp = RT_inv.dot(padded_points_camera)
#         Z = z
#         s = Z / tmp[0]
#         X = -(s * tmp[1])
#         Y = -(s * tmp[2])
#
#         # RtT = np.dot(R.T, T)
#         # tmp = np.dot(R.T, points_camera)
#         # Z = z
#         # s = (Z + RtT[0]) / tmp[0]
#         # X = -(s * tmp[1] - RtT[1])
#         # Y = -(s * tmp[2] - RtT[2])
#
#         points_3d = np.vstack((X, Y, Z))
#         padded_points_3d = np.vstack((points_3d, np.ones(points_3d.shape[1])))
#         # points_3d = np.dot(calibration.lidar_to_ego, padded_points_3d)
#         points_3d = np.dot(calibration.ego_to_lidar, padded_points_3d)
#         points_3d = points_3d[0:3]
#
#     return points_3d
