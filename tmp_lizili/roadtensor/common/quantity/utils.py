import os
import torch.nn as nn

from .quantity_layers import Identity


def merge_freezebn(model):
    conv_layer = None
    for name, layer in model.named_modules():
        # we assume the bn layer is defined right after the conv layer.
        if type(layer).__name__ == 'Conv2d' and layer.bias is None:
            assert conv_layer is None, "Please put bn right after the conv in your __init__()."
            conv_layer = layer
        if type(layer).__name__ == 'FrozenBatchNorm2d':
            assert conv_layer is not None, "Please put bn right after the conv in your __init__()."
            assert conv_layer.bias is None, \
                "The conv {} before freezebn {} has bias.".format(conv_layer, layer)

            conv_layer.weight = nn.Parameter(conv_layer.weight * layer.weight.view(-1, 1, 1, 1))
            conv_layer.bias = nn.Parameter(layer.bias)

            name_list = name.split('.')[:-1]
            module = model
            for n in name_list:
                module = getattr(module, n)
            module.add_module(name.split('.')[-1], Identity())
            conv_layer = None
    return model


def walk_dirs(dir_name, file_type=None):
    files_path = []

    for root, dir, files in os.walk(dir_name):
        for name in files:
            file_path = root + "/" + name
            if file_type and not file_path.endswith(file_type):
                continue
            files_path.append(file_path)

    return files_path
