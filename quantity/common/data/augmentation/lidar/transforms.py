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
import numpy as np
import torch


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, inputs, targets):
        for t in self.transforms:
            inputs, targets = t(inputs, targets)
        return inputs, targets

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class RandomHorizontalFlip(object):
    def __init__(self, probability=0.5):
        self.__probability = probability
        pass

    def __call__(self, inputs, targets):
        """
        :param inputs: xyzd point in ego coordinate. [N, 4]
        :param targets: ground true boxes, [x, y, z, w, l, h, ry], [N, 7]
        :return:
        """
        points = inputs["points"]
        gt_boxes = targets["gt_boxes"]
        enable = np.random.choice(
            [False, True], replace=False, p=[1 - self.__probability, self.__probability])
        if enable:
            gt_boxes[:, 1] = -gt_boxes[:, 1]
            gt_boxes[:, 6] = -gt_boxes[:, 6]
            points[:, 1] = -points[:, 1]
        return inputs, targets


class RandomScaling(object):
    def __init__(self, scale=0.05):
        self.__scale = scale
        if not isinstance(self.__scale, list):
            self.__scale = [-self.__scale, self.__scale]

    def __call__(self, inputs, targets):
        """
        :param inputs: xyzd point in ego coordinate. [N, 4]
        :param targets: ground true boxes, [x, y, z, w, l, h, ry], [N, 7]
        :return:
        """
        points = inputs["points"]
        gt_boxes = targets["gt_boxes"]

        noise_scale = np.random.uniform(self.__scale[0] + 1, self.__scale[1] + 1)
        points[:, :3] *= noise_scale
        gt_boxes[:, :6] *= noise_scale
        return inputs, targets


class RandomRotation(object):
    def __init__(self, rotation_angle=np.pi / 4):
        self.__rotation = rotation_angle
        if not isinstance(self.__rotation, list):
            self.__rotation = [-self.__rotation, self.__rotation]

    def __rotation_points_single_angle(self, points, angle, axis=0):
        # points: [N, 3]
        rot_sin = np.sin(angle)
        rot_cos = np.cos(angle)
        if axis == 1:
            rot_mat_t = np.array(
                [[rot_cos, 0, -rot_sin], [0, 1, 0], [rot_sin, 0, rot_cos]],
                dtype=points.dtype)
        elif axis == 2 or axis == -1:
            rot_mat_t = np.array(
                [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]],
                dtype=points.dtype)
        elif axis == 0:
            rot_mat_t = np.array(
                [[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]],
                dtype=points.dtype)
        else:
            raise ValueError("axis should in range")

        return points @ rot_mat_t

    def __call__(self, inputs, targets):
        """
        :param inputs: xyzd point in ego coordinate. [N, 4]
        :param targets: ground true boxes, [x, y, z, w, l, h, ry], [N, 7]
        :return:
        """
        points = inputs["points"]
        gt_boxes = targets["gt_boxes"]

        noise_rotation = np.random.uniform(self.__rotation[0], self.__rotation[1])

        points[:, :3] = self.__rotation_points_single_angle(
            points[:, :3], noise_rotation, axis=2)
        gt_boxes[:, :3] = self.__rotation_points_single_angle(
            gt_boxes[:, :3], noise_rotation, axis=2)
        gt_boxes[:, 6] += noise_rotation
        return inputs, targets


class ToTensor(object):
    def __call__(self, inputs, targets):
        # see https://stackoverflow.com/questions/48482787/pytorch-memory-model-torch-from-numpy-vs-torch-tensor
        # we prefer torch.from_numpy over torch.Tensor
        for k in inputs:
            inputs[k] = torch.from_numpy(inputs[k])
        for k in targets:
            targets[k] = torch.from_numpy(targets[k])
        return inputs, targets
