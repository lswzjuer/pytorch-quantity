import pickle

import numpy as np

from roadtensor.common.data.augmentation.lidar import (RandomRotation, RandomHorizontalFlip, RandomScaling)
from roadtensor.common.data.io import (read_lidar_pointcloud, read_3d_label)


def convert_obstacle_to_gt_boxes(obstacle_list):
    return np.array(
        [[obj.x, obj.y, obj.z, obj.width, obj.length, obj.height, obj.rotation]
         for obj in obstacle_list]
    )


class TestLidarAugmentation(object):
    raw_dataset = pickle.load(open('/private/xuguodong/test_40.pkl', 'rb'))
    sample = raw_dataset["02_22-181215155652_491-0000"]
    lidar_frame_label_filename = sample['LidarObstaclesLabeled']

    pc = read_lidar_pointcloud(sample["MergePointCloud"])
    obstacle_list = read_3d_label(lidar_frame_label_filename)

    gt_boxes = convert_obstacle_to_gt_boxes(obstacle_list)

    def test_random_rotation(self):
        op = RandomRotation(rotation_angle=[-0.1 * np.pi, -0.25 * np.pi])
        pc_, gt_boxes_ = op(TestLidarAugmentation.pc, TestLidarAugmentation.gt_boxes)

        assert pc_.shape == TestLidarAugmentation.pc.shape
        assert gt_boxes_.shape == TestLidarAugmentation.gt_boxes.shape

    def test_random_flip(self):
        op = RandomHorizontalFlip(probability=0.7)
        pc_, gt_boxes_ = op(TestLidarAugmentation.pc, TestLidarAugmentation.gt_boxes)
        assert pc_.shape == TestLidarAugmentation.pc.shape
        assert gt_boxes_.shape == TestLidarAugmentation.gt_boxes.shape

    def test_random_scaling(self):
        op = RandomScaling()
        pc_, gt_boxes_ = op(TestLidarAugmentation.pc, TestLidarAugmentation.gt_boxes)
        assert pc_.shape == TestLidarAugmentation.pc.shape
        assert gt_boxes_.shape == TestLidarAugmentation.gt_boxes.shape
