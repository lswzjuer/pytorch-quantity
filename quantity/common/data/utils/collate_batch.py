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
from roadtensor.components.structures.image_list import to_image_list
from collections import defaultdict

class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def list2dict(self, transposed_batch):
        result_dict = defaultdict(list)
        for target in transposed_batch:
            for k, v in target.items():
                result_dict[k].append(v)
        return result_dict

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        input_dict = self.list2dict(transposed_batch[0])
        if "img" in input_dict:
            input_dict["img"] = to_image_list(input_dict["img"], self.size_divisible)
        target_dict = self.list2dict(transposed_batch[1])
        img_ids = transposed_batch[2]
        return input_dict, target_dict, img_ids
