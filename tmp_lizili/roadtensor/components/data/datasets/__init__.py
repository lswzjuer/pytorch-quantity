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
from roadtensor.common.utils.imports import try_import
from .coco import COCODataset
from .voc import PascalVOCDataset
from roadtensor.common.data.utils.concat_dataset import ConcatDataset

__all__ = ["COCODataset", "ConcatDataset", "PascalVOCDataset"]

# task specific modules
Vision2DDataset = try_import("roadtensor.vision2d.data.datasets", "Vision2DDataset")
if Vision2DDataset is not None:
    __all__.append("Vision2DDataset")
