import argparse
import numpy as np
import os,sys
import yaml
import time
import cv2
import random
import torch
import importlib
import torch.nn as nn
import math

from collections import OrderedDict, defaultdict, Counter
from termcolor import colored
from rewriter import BiasReWriter
from common.config import cfg
from common.quantity import DistributionCollector, Quantizer, walk_dirs, merge_freezebn



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



class Quantity(object):
    def __init__(self,model):
        assert os.path.isfile("configs.yml"), "./configs.yml"
        assert os.path.isfile("user_configs.yml"), "./configs.yml"

        #read quantity config file
        with open("./configs.yml") as f:
            self.config = yaml.load(f)

        #read user config file
        with open("./user_configs.yml") as f:
            self.user_config = yaml.load(f)

        #create output folders
        self.init_dir()
            
        net_path = self.user_config['PATH']['MODEL_NET_PATH']    #net path (.py)
        pth_path = self.user_config['PATH']['MODEL_PATH']  #weight path (.pth)
        cali_data_dir = self.user_config['PATH']['DATA_PATH']

        input_shape = self.user_config['MODEL']['INPUT_SHAPE'].split(',')
        
        module_list = net_path.split('/')
        module_net = module_list[1] + '.' + module_list[2][:-3]
        param = importlib.import_module(module_net)
        
        self.model = model
        self.model.load_state_dict(torch.load(pth_path))
        self.input_size = list(map(int, input_shape))
        
        self.net_info = self.build_net_structure(self.model)

    def build_net_structure(self, model):
        """
        intput : 网络的model
        output:
                获取模型的信息，将不关心的层过滤掉，构建模型结果net_info:{intput, type}

        """
        all_op_type = ['Conv2d', 'Linear', 'Eltwise', 'Concat', 'MaxPool2d', 'ReLU',
                    'UpsamplingNearest2d', 'myView']
        cared_op_type = ['Conv2d', 'Linear', 'Eltwise', 'Concat']
        allow_same_tid_op_type = ['ReLU', 'UpsamplingNearest2d', 'myView']

        net = OrderedDict()
        name_to_input_id = defaultdict(list)
        name_to_id = OrderedDict()
        name_to_type = {}
        id_to_name = {}
        hooks = []

        """
        name_to_input_id
        name_to_id
        name_to_type 
        (通过一次forward，注册hook)
        """
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

        #input_var = torch.rand(1, 3, 256, 256)
        input_var = torch.rand(*self.input_size)
        print('input shape:', input_var.shape)
        
        _ = model(input_var)
        print('forward end')
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
        # print('**********************************************')
        # print('name_to_id',name_to_id)
        # print('name_to_input_id',name_to_input_id)
        # print('name_to_type',name_to_type)
        # print('**********************************************')

        if len(name_to_id) != len(set(name_to_id.values())):
            repeat_id = [item for item, count in Counter(name_to_id.values()).items() if count > 1]
            rid_to_name = defaultdict(list)
            for name, id in name_to_id.items():
                if id in repeat_id:
                    rid_to_name[id].append(name)
            print(name_to_id, '\n\n', rid_to_name)
            raise AssertionError("Some layers returned same tensor.")
        #print('**********************************************')

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

        net = self.prune_net_info(net, keep_node_list)
        return net

    def prune_net_info(self, net_info, keep_node_list):
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
    
    def preprocess(self, image):
        if (self.user_config['PRE_PROCESS']['IMG']):
            mean = slef.user_config['PRE_PROCESS']['IMG_SET']['MEAN']
            resize = tuple(map(float, slef.user_config['PRE_PROCESS']['IMG_SET']['RESIZE'].split(',')))
            scale = slef.user_config['PRE_PROCESS']['IMG_SET']['SCALE']
            origimg = cv2.imread(image_path)
            if origimg is None:
                return False
            origimg = cv2.resize(origimg, resize)
            img = origimg.astype(np.float32) - mean
            img = img.transpose((2, 0, 1))
            img = img[np.newaxis, :]
            img = torch.tensor(img, dtype=torch.float).cuda()
            return img
        else:
            return image

    def net_forward(self, net, image_path):
     
        img = self.preprocess(image_path)
        start = time.clock()
        _ = net(img)
        end = time.clock()
        print("forward time : %.3f s" % (end - start))

    def get_merge_groups(self, net):
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

    def activation_quantize(self, images_files):
        interval_num = self.config['SETTINGS']['INTERVAL_NUM']
        statistic = self.config['SETTINGS']['STATISTIC']
        worker_num = self.config['SETTINGS']['WORKER_NUM']
        table_file = self.config['OUTPUT']['FEAT_BIT_TABLE']

        table_file_content = []
        bottom_feat_names = self.get_merge_groups(self.net_info)
        top_feat_names = ['image'] + list(self.net_info.keys())

        max_vals = {}
        distribution_intervals = {}
        for i, feat_name in enumerate(top_feat_names):
            max_vals[feat_name] = 0
            distribution_intervals[feat_name] = 0

        collector = DistributionCollector(top_feat_names, interval_num=interval_num,
                                        statistic=statistic, worker_num=worker_num,
                                        debug=False)
        quantizer = Quantizer(top_feat_names, worker_num=worker_num, debug=False)

        named_feats, _ = self.hook_model(self.model)
        # run float32 inference on calibration dataset to find the activations range
        for i, image in enumerate(images_files):
            print('input size:',image.shape)
            self.net_forward(self.model, image)
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
            self.net_forward(self.model, image)
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
            if feat_name not in self.net_info:
                assert feat_name == 'image', feat_name
            elif is_first_op:
                feat_str = feat_str + ' ' + str(bits['image'])
                is_first_op = False
            elif len(self.net_info[feat_name]['inputs']) > 0:
                for inp_name in self.net_info[feat_name]['inputs']:
                    feat_str = feat_str + ' ' + str(bits[inp_name])
            else:
                raise NotImplementedError(self.net_info[feat_name])

            table_file_content.append(feat_str)

        with open(table_file, 'w') as f:
            for tabel_line in table_file_content:
                f.write(tabel_line + "\n")

    def hook_model(self, model):
        
        """
        创建hook获取每层的结果
        """

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




    def init_dir(self):
        """
        ctreate the output folders if they don't exist
        """
        work_dir = self.config['OUTPUT']['WORK_DIR']
        weight_dir = self.config['OUTPUT']['WEIGHT_DIR']
        bias_dir = self.config['OUTPUT']['BIAS_DIR']

        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)
        if not os.path.exists(bias_dir):
            os.makedirs(bias_dir)

    def rewrite_weight(self):

        weight_dir = self.config['OUTPUT']['WEIGHT_DIR']
        bias_dir = self.config['OUTPUT']['BIAS_DIR']
        output_weight_dir = self.config['OUTPUT']['FINAL_WEIGHT_DIR']
        output_bias_dir = self.config['OUTPUT']['FINAL_BIAS_DIR']
        weight_table = self.config['OUTPUT']['WEIGHT_BIT_TABLE']
        feat_table = self.config['OUTPUT']['FEAT_BIT_TABLE']
        max_shift_limit = self.config['SETTINGS']['MAX_SHIFT']

        rewriter = BiasReWriter(weight_dir, bias_dir, output_weight_dir, output_bias_dir,
                                weight_table, feat_table, max_shift_limit=max_shift_limit)

        # weight tabel
        weight_bits, bias_bits = rewriter.get_weight_info()
        feat_bits, infeat_bits = rewriter.get_feat_info()

        empty_set = (set(bias_bits.keys()).difference(set(feat_bits.keys())).union(
            set(feat_bits.keys()).difference(set(bias_bits.keys()))
        ))
        if empty_set != set():
            print(colored("These layers not include params but we care about their features:",
                        'red'), empty_set)

        print(colored("Align bias bit:", 'cyan'))
        # now, bias_bits == feat_bits == new_bias.
        rewriter.rewrite_bias_table(bias_bits, feat_bits)
        rewriter.rewrite_bias_dir(bias_bits, feat_bits)

        print(colored("Add max shift limitation:", 'cyan'))
        need_flag ,new_weight = rewriter.max_shift_limit_weight(feat_bits, infeat_bits, weight_bits)
        if need_flag :
            print('rewirte weight!!!!')
            # now, weight_bits <= max_shift + output_bits - input_bits.
            rewriter.rewrite_weight_table(weight_bits, new_weight)
            rewriter.rewrite_weight_dir(weight_bits, new_weight)

        print("Done!")

    def weight_quantize(self):
        """
        Weight quantize for Pytorch.
        Supported op: Conv2D or Conv2D-like ops.

        Args:
            model: nn.Module in Pytorch.
        """
        interval_nun = self.config['SETTINGS']['INTERVAL_NUM']
        statistic = self.config['SETTINGS']['STATISTIC']
        worker_num = self.config['SETTINGS']['WORKER_NUM']
        support_dilation = self.config['SETTINGS']['SUPPORT_DILATION']
        weight_dir = self.config['OUTPUT']['WEIGHT_DIR']
        bias_dir = self.config['OUTPUT']['BIAS_DIR']
        table_file = self.config['OUTPUT']['WEIGHT_BIT_TABLE']

        param_names = []
        param_shapes = {}
        params = {}
        tabel_file = []

        for name, param in self.model.named_parameters():
            if not name.endswith('weight') and not name.endswith('bias'):
                print(colored("[WARNING]", 'red'), " not supported param: {}".format(name))
                continue

            name_list = name.split('.')[:-1]
            module = self.model
            for n in name_list:
                module = getattr(module, n)

            param_np = param.detach().numpy()

            if name.endswith('weight') and isinstance(module, nn.Conv2d):
                dilation = module.dilation
                if dilation != (1, 1) and not support_dilation:
                    param_np = self.dilation_to_zero_padding(param_np, dilation)

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

        self.rewrite_weight()

    def dilation_to_zero_padding(self, tensor, dilation):
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



  