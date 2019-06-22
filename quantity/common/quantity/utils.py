import os
import torch.nn as nn
import torch
from .fabu_layer import Identity
from termcolor import colored

def merge_bn(model,device = 'cpu'):

    conv_layer = None
    for name, layer in model.named_modules():
        # we assume the bn layer is defined right after the conv layer.
        if type(layer).__name__ == 'Conv2d':
            #assert conv_layer is None, "Please put bn right after the conv in your __init__()."
            conv_layer = layer
        if type(layer).__name__ == 'BatchNorm2d':
            assert conv_layer is not None, "Please put bn right after the conv in your __init__()."
            # assert conv_layer.bias is None, "The conv {} before freezebn {} has bias.".format(conv_layer, layer)

            # Propose BN parameter
            alpha=layer.weight.data
            beta=layer.bias.data
            var=layer.running_var
            mean=layer.running_mean

            # Propose conv layer parameter
            weight=conv_layer.weight
            bias=conv_layer.bias

            assert type(weight).__name__!="NoneType", "The conv weight can`t be None"
            weight_data=weight.data
            if type(bias).__name__=="NoneType":
                bias_data=torch.zeros(size=[conv_layer.out_channels])
            else:
                bias_data=bias.data

            # merge bn to conv layer
            tmp = alpha / torch.sqrt(var + 1e-5)
            if (device == 'cuda'):
                tmp = tmp.cuda()
                bias_data = bias_data.cuda()
            new_weight = tmp.view(tmp.size()[0], 1, 1, 1)*weight_data
            new_bias=tmp*(bias_data -mean ) + beta

            # Modify convolution layer parameters
            conv_layer.weight = nn.Parameter(new_weight)
            conv_layer.bias = nn.Parameter(new_bias)

            # Replace the BN layer with the Identity layer
            find_module=model
            name_split = name.split('.')
            if len(name_split)==1:
                find_module.add_module(name_split[0],Identity())
                print(colored('The layer change: {} ==>Identity'.format(name),'green'))
            elif len(name_split)>=2:
                father_list=name_split[:-1]
                for n in father_list:
                    find_module=getattr(find_module,n)
                find_module.add_module(name_split[-1], Identity())
                print(colored('The layer change: {} ==>Identity'.format(name),'green'))
            else:
                raise ValueError("the layer name is wrong")

            conv_layer=None

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


def tid(tensor):
    """
    Return unique id for the tensor based on the tensor value.

    Args:
        tensor: torch tensor of any shape.

    Returns:
        str
    """
    ids = []
    x = tensor.cpu()
    ids.append(
        str(int((x[..., 0].max() + x[..., 0].min()).item() * 1e4 % 1e4)))
    ids.append(str(int((x.max() + x.min()).item() * 1e4 % 1e4)))
    ids.append(str(int(x[..., 0].mean().item() * 1e4 % 1e4)))
    ids.append(str(int(x.mean().item() * 1e4 % 1e4)))
    return ''.join(ids)
