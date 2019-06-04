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
"""
Implements the Generalized R-CNN framework
"""
import io
from loguru import logger

import torch
from torch import nn

from roadtensor._C import Engine
from roadtensor.components.structures.image_list import to_image_list

from roadtensor.components.modeling.backbone import build_backbone
from roadtensor.components.modeling.anchor_heads.builder import build_retinanet_head

class RetinaNet(nn.Module):
    """
    RetinaNet
    
    It consists of three main parts:
    - backbone
    - fpn
    - retina head: takes the features and perform classification and regress bboxes
    """

    def __init__(self, cfg):
        super(RetinaNet, self).__init__()

        # FIXME no used
        self.exporting = False
        
        self.backbone = build_backbone(cfg)
        self.head = build_retinanet_head(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        # print("images shape {}".format(images.image_sizes))
        images = to_image_list(images)
        features = self.backbone(images.tensors)

        return self.head(images, features, targets)

    def export(self, size, batch, precision, calibration_files, calibration_table, onnx_only=False):
        import torch.onnx.symbolic

        # Override Upsample's ONNX export until new opset is supported
        # @torch.onnx.symbolic.parse_args('v', 'is')
        # def upsample_nearest2d(g, input, output_size):
        #     height_scale = float(output_size[-2]) / input.type().sizes()[-2]
        #     width_scale = float(output_size[-1]) / input.type().sizes()[-1]
        #     return g.op("Upsample", input,
        #         scales_f=(1, 1, height_scale, width_scale),
        #         mode_s="nearest")
        # torch.onnx.symbolic.upsample_nearest2d = upsample_nearest2d

        # Export to ONNX
        logger.info('Exporting to ONNX...')
        self.head.exporting = True
        onnx_bytes = io.BytesIO()
        zero_input = torch.zeros([batch, *size]).cuda()
        torch.onnx.export(self.cuda(), zero_input, onnx_bytes)
        self.head.exporting = False

        if onnx_only:
            return onnx_bytes.getvalue()
