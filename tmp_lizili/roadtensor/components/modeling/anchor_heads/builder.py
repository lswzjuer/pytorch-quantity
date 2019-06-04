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
import torch

from roadtensor.components.modeling import registry

from .retina_head.retina_head import RetinaNetHead
from .rpn_head.rpn_head import RPNHead

def build_anchor_head(cfg, in_channels):
    anchor_head = registry.ANCHOR_HEADS[cfg.MODEL.ANCHOR_HEAD]
    return anchor_head(cfg, in_channels)

def build_rpn_head(cfg, in_channels):
    return RPNHead(cfg, in_channels)

def build_retinanet_head(cfg, in_channels):
    return RetinaNetHead(cfg, in_channels)