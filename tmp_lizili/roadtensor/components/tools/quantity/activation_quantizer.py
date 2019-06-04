import argparse
import numpy as np
import os
import yaml
import time
import cv2
import random
import torch
from collections import OrderedDict, defaultdict, Counter
from termcolor import colored

from roadtensor.common.config import cfg
from roadtensor.common.utils.checkpoint import DetectronCheckpointer
from roadtensor.common.quantity import DistributionCollector, Quantizer, walk_dirs, merge_freezebn
from roadtensor.components.modeling.models.detector import build_detection_model


def net_forward(net, image_path):
    print(image_path, end=' ')
    origimg = cv2.imread(image_path)
    if origimg is None:
        return False

    origimg = cv2.resize(origimg, (1280, 640))

    mean = (102.9801, 115.9465, 122.7717)
    img = origimg.astype(np.float32) - mean
    img = img.transpose((2, 0, 1))
    img = img[np.newaxis, :]
    img = torch.tensor(img, dtype=torch.float).cuda()

    start = time.clock()
    _ = net(img)
    end = time.clock()
    print("forward time : %.3f s" % (end - start))


def get_merge_groups(net):
    merge_ops = ['Eltwise', 'Concat']
    cared_op_type = ['Conv2d', 'Linear']

    merge_layer = []
    for name, info in net.items():
        if info['type'] in merge_ops:
            merge_layer.append(name)

    merge_layer.reverse()
    print(colored('merge layers:', 'green'), merge_layer)

    vis = []

    def _dfs(name):
        if name in vis:
            return []
        vis.append(name)

        info = net[name]
        bottoms = info['inputs']
        names = []

        if len(bottoms) == 0:
            return []

        for bottom in bottoms:
            if net[bottom]['type'] not in cared_op_type:
                names.extend(_dfs(bottom))
            else:
                names.append(bottom)
        return names

    merge_groups = []
    for layer_name in merge_layer:
        b_names = _dfs(layer_name)
        print(colored(layer_name, 'green'), b_names)
        if b_names:
            merge_groups.append(b_names)

    return merge_groups


def activation_quantize(net, net_info, images_files, cfg):
    interval_num = cfg['SETTINGS']['INTERVAL_NUM']
    statistic = cfg['SETTINGS']['STATISTIC']
    worker_num = cfg['SETTINGS']['WORKER_NUM']
    table_file = cfg['OUTPUT']['FEAT_BIT_TABLE']

    table_file_content = []
    bottom_feat_names = get_merge_groups(net_info)
    top_feat_names = ['image'] + list(net_info.keys())

    max_vals = {}
    distribution_intervals = {}
    for i, feat_name in enumerate(top_feat_names):
        max_vals[feat_name] = 0
        distribution_intervals[feat_name] = 0

    collector = DistributionCollector(top_feat_names, interval_num=interval_num,
                                      statistic=statistic, worker_num=worker_num,
                                      debug=False)
    quantizer = Quantizer(top_feat_names, worker_num=worker_num, debug=False)

    named_feats, _ = hook_model(net)
    # run float32 inference on calibration dataset to find the activations range
    for i, image in enumerate(images_files):
        net_forward(net, image)
        print("loop stage 1 : %d" % (i))
        # find max threshold
        tensors = {}
        for name, feat in named_feats.items():
            tensors[name] = feat.flatten()
        collector.refresh_max_val(tensors)

    print(colored('max_vals', 'green'), collector.max_vals)
    distribution_intervals = collector.distribution_intervals
    for b_names in bottom_feat_names:
        assert len(b_names) > 1
        tmp_distribution_interval = 0
        # distributions
        for pre_feat_name in b_names:
            tmp_distribution_interval = max(tmp_distribution_interval,
                                            distribution_intervals[pre_feat_name])
        for pre_feat_name in b_names:
            distribution_intervals[pre_feat_name] = tmp_distribution_interval

    # for each layer, collect histograms of activations
    print(colored("Collect histograms of activations:", 'cyan'))
    for i, image in enumerate(images_files):
        net_forward(net, image)
        print("loop stage 2 : %d" % (i))
        start = time.clock()
        tensors = {}
        for name, feat in named_feats.items():
            tensors[name] = feat.flatten()
        collector.add_to_distributions(tensors)
        end = time.clock()
        print("add cost %.3f s" % (end - start))

    distributions = collector.distributions

    # refresh the distribution of the bottom feat of layers like concat and eltwise.
    for b_names in bottom_feat_names:
        assert len(b_names) > 1
        tmp_distributions = np.zeros(interval_num)
        # distributions
        for pre_feat_name in b_names:
            tmp_distributions += distributions[pre_feat_name]
        for pre_feat_name in b_names:
            distributions[pre_feat_name] = tmp_distributions

    quantizer.quantize(distributions, distribution_intervals)
    bits = quantizer.bits

    is_first_op = True
    for feat_name in top_feat_names:
        feat_str = feat_name.replace('.', '_') + " " + str(bits[feat_name])
        if feat_name not in net_info:
            assert feat_name == 'image', feat_name
        elif is_first_op:
            feat_str = feat_str + ' ' + str(bits['image'])
        elif len(net_info[feat_name]['inputs']) > 0:
            for inp_name in net_info[feat_name]['inputs']:
                feat_str = feat_str + ' ' + str(bits[inp_name])
        else:
            raise NotImplementedError(net_info[feat_name])

        table_file_content.append(feat_str)

    with open(table_file, 'w') as f:
        for tabel_line in table_file_content:
            f.write(tabel_line + "\n")


def hook_model(model):
    out_feat = OrderedDict()
    hooks = []

    def _make_hook(name, is_input):
        def hook(m, input, output):
            out_feat[name] = output.detach().cpu().numpy()
            if is_input:
                out_feat['image'] = input[0].detach().cpu().numpy()

        return hook

    is_input = True
    for i, (name, m) in enumerate(model.named_modules()):
        if type(m).__name__ in ['Conv2d', 'Linear', 'ReLU', 'Eltwise', 'Concat']:
            hook = m.register_forward_hook(_make_hook(name, is_input))
            hooks.append(hook)
            is_input = False

    return out_feat, hooks


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
    ids.append(str(int((x[..., 0].max() + x[..., 0].min()).item() * 1e4 % 1e4)))
    ids.append(str(int((x.max() + x.min()).item() * 1e4 % 1e4)))
    ids.append(str(int(x[..., 0].mean().item() * 1e4 % 1e4)))
    ids.append(str(int(x.mean().item() * 1e4 % 1e4)))
    return ''.join(ids)


def prune_net_info(net_info, keep_node_list):
    all_op_names = set(list(net_info.keys()))
    prune_op_names = all_op_names - set(keep_node_list)
    keys = list(net_info.keys())[::-1]
    net_info_rev, net_info_new = OrderedDict(), OrderedDict()
    for k in keys:
        net_info_rev[k] = net_info[k]

    def _dfs(name):
        """
        Start from prune op, the input length must be 1.
        """
        info = net_info_rev[name]
        assert len(info['inputs']) <= 1, (info['inputs'], name)
        if len(info['inputs']) == 0:
            return None

        inp = info['inputs'][0]

        if inp in prune_op_names:
            return _dfs(inp)
        else:
            return inp

    for name, info in net_info_rev.items():
        if name in prune_op_names:
            continue

        for ind in range(len(info['inputs'])):
            inp = info['inputs'][ind]
            if inp in prune_op_names:
                info['inputs'][ind] = _dfs(inp)

    for k in net_info.keys():
        if k not in prune_op_names:
            net_info_new[k] = net_info_rev[k]

    return net_info_new


def build_net_structure(model):
    all_op_type = ['Conv2d', 'Linear', 'Eltwise', 'Concat', 'MaxPool2d', 'ReLU',
                   'UpsamplingNearest2d']
    cared_op_type = ['Conv2d', 'Linear', 'Eltwise', 'Concat']
    allow_same_tid_op_type = ['ReLU', 'UpsamplingNearest2d']

    net = OrderedDict()
    name_to_input_id = defaultdict(list)
    name_to_id = OrderedDict()
    name_to_type = {}
    id_to_name = {}
    hooks = []

    def _make_hook(name):
        def _hook(m, input, output):
            layer_type = type(m).__name__

            for t in input:
                name_to_input_id[name].append(tid(t))

            if name not in name_to_id:
                name_to_type[name] = layer_type
                name_to_id[name] = tid(output)
            else:
                # share-param structure
                raise NotImplementedError

        return _hook

    for name, m in model.named_modules():
        if type(m).__name__ in all_op_type:
            hook = m.register_forward_hook(_make_hook(name))
            hooks.append(hook)

    input_var = torch.rand(1, 3, 640, 320).cuda()
    _ = model(input_var)
    for hook in hooks:
        hook.remove()

    # since we use OrderDict, so we can just generate new id for repeat tensor and
    # modify the following ids to new one.
    modify_id = {}
    for name, idx in name_to_id.items():
        for i, inp_idx in enumerate(name_to_input_id[name]):
            if inp_idx in modify_id:
                name_to_input_id[name][i] = modify_id[inp_idx]

        for i, inp_idx in enumerate(name_to_input_id[name]):
            if inp_idx == idx:
                if name_to_type[name] in allow_same_tid_op_type:
                    rand_int = random.randint(0, 1e9)
                    new_idx = str(int(idx) + rand_int)
                    name_to_id[name] = new_idx
                    modify_id[idx] = new_idx
                else:
                    raise ValueError("Same input and output id, the op {} is useful?".format(name))

    if len(name_to_id) != len(set(name_to_id.values())):
        repeat_id = [item for item, count in Counter(name_to_id.values()).items() if count > 1]
        rid_to_name = defaultdict(list)
        for name, id in name_to_id.items():
            if id in repeat_id:
                rid_to_name[id].append(name)
        print(name_to_id, '\n\n', rid_to_name)
        raise AssertionError("Some layers returned same tensor.")

    for n, i in name_to_id.items():
        id_to_name[i] = n

    trigger = None
    for i, (name, idx) in enumerate(name_to_id.items()):
        inputs = []
        for inp_idx in name_to_input_id[name]:
            if inp_idx in id_to_name:
                inputs.append(id_to_name[inp_idx])
            elif i != 0:
                trigger = name
        net[name] = {'inputs': inputs, 'type': name_to_type[name]}

    assert trigger is None, "Can't find the input tensor of {} \n {}".format(trigger, net)
    keep_node_list = []
    for name, info in net.items():
        if info['type'] in cared_op_type:
            keep_node_list.append(name)

    net = prune_net_info(net, keep_node_list)
    return net


def main():
    parser = argparse.ArgumentParser(
        description="Quantize activations of model and generate .tabel")
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
    max_cali_num = quantity_cfg['SETTINGS']['MAX_CALI_IMG_NUM']
    cali_data_dir = quantity_cfg['PATH']['DATA_PATH']

    model = build_detection_model(cfg)
    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    if is_merge_freezebn:
        model = merge_freezebn(model)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(quantity_cfg['SETTINGS']['GPU'])
    model.eval().cuda()

    file_names = walk_dirs(cali_data_dir, file_type='.jpg')[:max_cali_num]
    net_info = build_net_structure(model)
    activation_quantize(model, net_info, file_names, quantity_cfg)


if __name__ == "__main__":
    main()
