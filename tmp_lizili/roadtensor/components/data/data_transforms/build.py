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
from roadtensor.common.data.augmentation.image import transforms as image_transforms


def build_transforms(cfg, is_train=True):
    if "coco" in cfg.DATASETS.TRAIN[0] or "voc" in cfg.DATASETS.TRAIN[0] or "fabu2d" in cfg.DATASETS.TRAIN[0]:
        return build_vision2d_transforms(cfg, is_train)
    else:
        raise RuntimeError(
            "dataset type should be in voc, coco or lidar_fusion, got {}".format(
                cfg.DATASETS.TRAIN[0])
        )


def build_vision2d_transforms(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_prob = 0

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = image_transforms.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )

    transform = image_transforms.Compose(
        [
            image_transforms.Resize(min_size, max_size),
            image_transforms.RandomHorizontalFlip(flip_prob),
            image_transforms.ToTensor(),
            normalize_transform,
        ]
    )
    return transform
