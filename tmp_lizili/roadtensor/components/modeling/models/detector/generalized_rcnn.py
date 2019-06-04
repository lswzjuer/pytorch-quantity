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
from roadtensor.components.modeling.anchor_heads.builder import build_rpn_head
from roadtensor.components.modeling.roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.exporting = False
        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn_head(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

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
        # for i, feature in enumerate(features):
            # print("feature {} shape {}".format(i, feature.shape))

        proposals, proposal_losses = self.rpn(images, features, targets)

        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result

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
        self.rpn.exporting = True
        for i in range(len(self.roi_heads)):
            self.roi_heads[i].exporting = True
        onnx_bytes = io.BytesIO()
        zero_input = torch.zeros([batch, *size]).cuda()
        torch.onnx.export(self.cuda(), zero_input, onnx_bytes)
        self.rpn.exporting = False
        for i in range(len(self.roi_heads)):
            self.roi_heads[i].exporting = False

        if onnx_only:
            return onnx_bytes.getvalue()

        # Build TensorRT engine
        # model_name = '_'.join([k for k, _ in self.backbones.items()])
        # anchors = [generate_anchors(stride, self.ratios, self.scales).view(-1).tolist() 
            # for stride in self.strides]
        # return Engine(onnx_bytes.getvalue(), len(onnx_bytes.getvalue()), batch, precision,
            # self.threshold, self.top_n, anchors, self.nms, self.detections, calibration_files, model_name, calibration_table, verbose)
