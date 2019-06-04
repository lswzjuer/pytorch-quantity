import pickle
import os.path as osp
from roadtensor.common.data.io import (read_lidar_pointcloud, read_3d_label, read_img, read_calib_config,
                                       read_camera_calib, read_lidar_calib)


class TestIO(object):
    """docstring for TestIO"""
    raw_dataset = pickle.load(open('/private/xuguodong/test_40.pkl', 'rb'))
    sample = raw_dataset["02_22-181215155652_491-0000"]
    lidar_frame_label_filename = sample['LidarObstaclesLabeled']
    calib_dir = sample['calib_dir']

    def test_read_lidar_pointcloud(self):
        pc = read_lidar_pointcloud(TestIO.sample["MergePointCloud"])

    def test_read_3d_label(self):
        obstacle_list = read_3d_label(TestIO.lidar_frame_label_filename)

    def test_read_img(self):
        img = read_img(TestIO.sample['CameraHeadRight'], gray=False, rgb=True)

    def test_read_calib_config(self):
        calib_config = read_calib_config(
            TestIO.calib_dir, sensor=True, camera=True)

    def test_read_camera_calib(self):
        calib_config = read_calib_config(
            TestIO.calib_dir, sensor=True, camera=True)
        camera_calib = read_camera_calib(calib_config['CameraHeadRight'])

    def read_lidar_calib(self):
        calib_config = read_calib_config(
            TestIO.calib_dir, sensor=True, camera=True)
        lidar2imu = read_lidar_calib(calib_config['LidarMain'])
        ego2imu = read_lidar_calib(calib_config['EgoFront'])
