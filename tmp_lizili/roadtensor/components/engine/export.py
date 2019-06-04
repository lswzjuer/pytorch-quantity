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
import logging
import time
import os

import torch
from tqdm import tqdm

def export(model, cfg, export_model_path):
    input_size = (cfg.EXPORT.IMG_CHANNEL,
                  cfg.EXPORT.IMG_HEIGHT, cfg.EXPORT.IMG_WIDTH)

    calibration_files = []
    if cfg.EXPORT.INT8:
        # Get list of images to use for calibration
        if os.path.isdir(cfg.EXPORT.CALIBRATION_IMAGES):
            import glob
            file_extensions = ['.jpg', '.JPG',
                               '.jpeg', '.JPEG', '.png', '.PNG']
            for ex in file_extensions:
                calibration_files += glob.glob("{}/*{}".format(
                    cfg.EXPORT.CALIBRATION_IMAGES, ex), recursive=True)
            # Only need enough images for specified num of calibration batches
            if len(calibration_files) >= cfg.EXPORT.CALIBRATION_BATCHES * cfg.EXPORT.BATCH:
                calibration_files = calibration_files[:(
                    cfg.EXPORT.CALIBRATION_BATCHES * cfg.EXPORT.BATCH)]
            else:
                print('Only found enough images for {} batches. Continuing anyway...'.format(
                    len(calibration_files) // cfg.EXPORT.BATCH))

            random.shuffle(calibration_files)

    precision = "FP32"
    if cfg.EXPORT.INT8:
        precision = "INT8"
    elif not cfg.EXPORT.FULL_PRECISION:
        precision = "FP16"
    exported = model.export(input_size, cfg.EXPORT.BATCH, precision,
                 calibration_files, cfg.EXPORT.CALIBRATION_TABLE, onnx_only = cfg.EXPORT.ONNX_ONLY)
    if cfg.EXPORT.ONNX_ONLY:
        with open(export_model_path, 'wb') as out:
            out.write(exported)
