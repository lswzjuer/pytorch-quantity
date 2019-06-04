import pickle
from roadtensor.common.data.io import (read_lidar_pointcloud, read_calib_config,
                                       read_camera_calib, read_lidar_calib)
from roadtensor.common.data.calibration import get_calibration, transform_coordinate, project

class TestCalibration(object):

    raw_dataset = pickle.load(open('/private/xuguodong/test_40.pkl', 'rb'))
    sample = raw_dataset["02_22-181215155652_491-0000"]
    lidar_frame_label_filename = sample['LidarObstaclesLabeled']
    calib_dir = sample['calib_dir']

    def test_calibration(self):
        calibration = get_calibration(TestCalibration.calib_dir)

    def test_projection(self):
        calibration = get_calibration(TestCalibration.calib_dir)
        pts = read_lidar_pointcloud(TestCalibration.sample["MergePointCloud"])
        pts_2d = project(pts[:, 0:3].T, p=calibration.p)

    def test_transform_coordinate(self):
        calibration = get_calibration(TestCalibration.calib_dir)
        pts = read_lidar_pointcloud(TestCalibration.sample["MergePointCloud"])
        pts_imu = transform_coordinate(pts[:, 0:3].T, calibration.ego_to_imu)





