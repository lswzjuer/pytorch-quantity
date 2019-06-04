import argparse
import numpy as np
import math
import os
import json
import yaml
import torch.nn as nn
from termcolor import colored

from roadtensor.common.config import cfg
from roadtensor.common.utils.checkpoint import DetectronCheckpointer
from roadtensor.common.quantity import DistributionCollector, Quantizer, merge_freezebn
from roadtensor.components.modeling.models.detector import build_detection_model


def init_dir(cfg):
    weight_dir = cfg['OUTPUT']['WEIGHT_DIR']
    bias_dir = cfg['OUTPUT']['BIAS_DIR']

    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    if not os.path.exists(bias_dir):
        os.makedirs(bias_dir)


def weight_quantize(model, cfg):
    """
    Weight quantize for Pytorch.
    Supported op: Conv2D or Conv2D-like ops.

    Args:
        model: nn.Module in Pytorch.
        cfg: quantity config file.
    """
    interval_nun = cfg['SETTINGS']['INTERVAL_NUM']
    statistic = cfg['SETTINGS']['STATISTIC']
    worker_num = cfg['SETTINGS']['WORKER_NUM']
    support_dilation = cfg['SETTINGS']['SUPPORT_DILATION']
    weight_dir = cfg['OUTPUT']['WEIGHT_DIR']
    bias_dir = cfg['OUTPUT']['BIAS_DIR']
    table_file = cfg['OUTPUT']['WEIGHT_BIT_TABLE']

    param_names = []
    param_shapes = {}
    params = {}
    tabel_file = []

    for name, param in model.named_parameters():
        if not name.endswith('weight') and not name.endswith('bias'):
            print(colored("[WARNING]", 'red'), " not supported param: {}".format(name))
            continue

        name_list = name.split('.')[:-1]
        module = model
        for n in name_list:
            module = getattr(module, n)

        param_np = param.detach().numpy()

        if name.endswith('weight') and isinstance(module, nn.Conv2d):
            dilation = module.dilation
            if dilation != (1, 1) and not support_dilation:
                param_np = dilation_to_zero_padding(param_np, dilation)

        param_names.append(name)
        param_shapes[name] = param_np.shape
        params[name] = param_np.flatten()

    collector = DistributionCollector(param_names, interval_num=interval_nun,
                                      statistic=statistic, worker_num=worker_num)
    quantizer = Quantizer(param_names, worker_num=worker_num)

    collector.refresh_max_val(params)
    print(colored('max vals:', 'green'), collector.max_vals)

    collector.add_to_distributions(params)
    quantizer.quantize(collector.distributions, collector.distribution_intervals)

    for name, bit in quantizer.bits.items():
        param_quantity = np.around(params[name] * math.pow(2, bit))
        param_quantity = np.clip(param_quantity, -128, 127)
        param_file_name = name.replace('.', '_')

        tabel_line = param_file_name + " " + str(bit)
        tabel_file.append(tabel_line)

        content = param_quantity.reshape(param_shapes[name]).astype(np.int32).tolist()
        if name.endswith('weight'):
            with open(os.path.join(weight_dir, param_file_name + '.json'), 'w') as file:
                json.dump(content, file, indent=4)
        elif name.endswith('bias'):
            with open(os.path.join(bias_dir, param_file_name + '.json'), 'w') as file:
                json.dump(content, file, indent=4)
        else:
            raise NotImplementedError(name)

    with open(table_file, 'w') as f:
        for tabel_line in tabel_file:
            f.write(tabel_line + "\n")


def dilation_to_zero_padding(tensor, dilation):
    out_channel, in_channel = tensor.shape[:2]
    assert tensor.shape[2] == tensor.shape[3] and dilation == (2, 2), "Not support."

    kernel_size = tensor.shape[2]
    new_kernel_size = kernel_size * 2 - 1
    new_tensor = np.zeros((out_channel, in_channel, new_kernel_size, new_kernel_size),
                          dtype=np.float32)
    source_loc = np.array(range(kernel_size))
    trans_loc = source_loc * 2

    for x, new_x in zip(source_loc, trans_loc):
        for y, new_y in zip(source_loc, trans_loc):
            new_tensor[..., new_x, new_y] = tensor[..., x, y]
    return new_tensor


def main():
    parser = argparse.ArgumentParser(
        description="Quantize weight of model and generate .tabel, weight/ and bias/")
    parser.add_argument(
        "--config-file",
        default="/roadtensor/roadtensor/components/configs/rpn_R_50_C4_1x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        '--yml',
        default="/roadtensor/roadtensor/components/tools/quantity/configs.yml",
        dest='yml',
        help="path to the yml file"
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    assert os.path.isfile(args.yml), args.yml
    with open(args.yml) as f:
        quantity_cfg = yaml.load(f)

    is_merge_freezebn = quantity_cfg['SETTINGS']['MERGE_FREEZEBN']

    model = build_detection_model(cfg)
    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    if is_merge_freezebn:
        model = merge_freezebn(model)
    model.eval()

    init_dir(quantity_cfg)
    weight_quantize(model, quantity_cfg)


if __name__ == "__main__":
    main()
