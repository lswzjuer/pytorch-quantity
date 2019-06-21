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
import json
from collections import OrderedDict, defaultdict, Counter
from termcolor import colored
from rewriter import BiasReWriter

sys.path.insert(0,'../')
from common.quantity import DistributionCollector, Quantizer, walk_dirs, merge_bn
from common.utils import tid


class Quantity(object):

    def __init__(self,model):

        assert os.path.isfile("configs.yml"), "./configs.yml"
        assert os.path.isfile("../test/user_configs.yml"), "../test/user_configs.yml"

        #read quantity config file
        with open("./configs.yml") as f:
            self.config = yaml.load(f)

        #read user config file
        with open("../test/user_configs.yml") as f:
            self.user_config = yaml.load(f)

        #create output folders
        self.init_dir()
            
        net_path = self.user_config['PATH']['MODEL_NET_PATH']    #net path (.py)
        pth_path = self.user_config['PATH']['MODEL_PATH']  #weight path (.pth)
        cali_data_dir = self.user_config['PATH']['DATA_PATH']

        input_shape = self.user_config['MODEL']['INPUT_SHAPE'].split(',')
        self.device = self.user_config['SETTINGS']['DEVICE']
        if (self.device == 'gpu'):
            gpu_id = self.user_config['SETTINGS']['GPU']
            torch.cuda.set_device(gpu_id)

        # define the support ops and special ops
        self._cared_op_type = self.config['SETTINGS']['CARE_OP_TYPE']
        self._all_op_type = self.config['SETTINGS']['ALL_OP_TYPE']
        self._allow_same_tid_op_type = self.config['SETTINGS']['ALLOW_SAME_TID_OP_TYPE']
        self._merge_op_type = self.config['SETTINGS']['MERGE_OP_YTPE']
        self._max_img_num = self.config['SETTINGS']['MAX_CALI_IMG_NUM']
        print(colored('max_img_num', 'green'), self._max_img_num)
        self.model = model
        #self.model.load_state_dict(torch.load(pth_path))
        self.input_size = tuple(map(int, input_shape))
        self.layers_num = 0
        self.name_to_param = OrderedDict()
        self.net_info = self.build_net_structure(self.model, self.input_size, self.device)
        
        self.cared_op_layer_names = self.get_cared_op_names(self.model)
        self._DKL_weight = False 

    
    def build_net_structure(self, model, input_size, device='cpu'):
        """
        intput : 网络的model
        output:
                获取模型的信息，将不关心的层过滤掉，构建模型结果net_info:{intput, type}

        """
        # all_op_type = ['Conv2d', 'Linear', 'Eltwise', 'Concat', 'MaxPool2d', 'ReLU',
        #                'UpsamplingNearest2d', 'View', 'AvgPool2d']
        #cared_op_type = ['Conv2d', 'Linear', 'Eltwise', 'Concat']
        cared_op_type = self._cared_op_type
        all_op_type = self._all_op_type

        allow_same_tid_op_type = self._allow_same_tid_op_type

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
        def register_hook(m):
            
            def _hook(m, input, output):
                
                layer_type = type(m).__name__
                class_name = str(m.__class__).split(".")[-1].split("'")[0]
                name_keys = "%s_%i" % (class_name, len(name_to_type) + 1)
              
                for t in input:
                    # print('input', tid(t))
                    name_to_input_id[name_keys].append(tid(t))

                if name_keys not in name_to_id.keys():
                    # print('output',tid(output))
                    name_to_type[name_keys] = layer_type
                    name_to_id[name_keys] = tid(output)
                else:
                    # share-param structure
                    raise NotImplementedError

            layer_type = type(m).__name__
            
            if (layer_type in all_op_type):
                hooks.append(m.register_forward_hook(_hook))


        device = device.lower()
        assert device in [
            "gpu",
            "cpu",
        ], "Input device is not valid, please specify 'cuda' or 'cpu'"

        if device == "gpu" and torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor

        # multiple inputs to the network
        if isinstance(input_size, tuple):
            input_size = [input_size]

        # batch_size of 1 for batchnorm
        x = [torch.rand(*input_size).type(dtype) for in_size in input_size]
  
        # register hook
        model.apply(register_hook)

        # make a forward pass
        # print(x.shape)
        model(*x)

        # remove these hooks
        for h in hooks:
            h.remove()
        
        # for para, names in name_to_param.items():
        #     print(para,names)


        self.layers_num = len(name_to_id)

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
                        print('name %s conflict! %s ' %
                              (name, str(idx)))
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
            #print(name_to_id, '\n\n', rid_to_name)
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

    def get_cared_op_names(self, model):
        layer_names = []
        for name, module in model.named_modules():
            if (type(module).__name__ in self._cared_op_type):
                layer_names.append(name)
        return layer_names



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
        data_option = int(self.user_config['PRE_PROCESS']['IMG'])
        if (data_option== 0):
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
        
        elif(data_option == 1):
            img, _ = image
            return img
        elif (data_option == 2):
            np_img = np.load(image)
            img = torch.tensor(np_img)
            img = img.view(1, *img.shape)
            return img

        else:
            print(colored("input option set wrong:",
                        'red'), data_option)



    def net_forward(self, net, image_path):
     
        img = self.preprocess(image_path)
        if (self.device == 'gpu'):
            img = img.cuda()
        start = time.clock()
        _ = net(img)
        end = time.clock()
        print("forward time : %.3f s" % (end - start))

    def get_merge_groups(self, net):
        merge_ops = self._merge_op_type
        cared_op_type = self._cared_op_type

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
        for top_name in top_feat_names:
            print(top_name)
        max_vals = {}
        distribution_intervals = {}
        for i, feat_name in enumerate(top_feat_names):
            max_vals[feat_name] = 0
            distribution_intervals[feat_name] = 0

        collector = DistributionCollector(top_feat_names, interval_num=interval_num,
                                        statistic=statistic, worker_num=worker_num,
                                        debug=False)
        quantizer = Quantizer(top_feat_names, worker_num=worker_num, debug=False)

        named_feats, _ = self.regist_hook_outfeature(self.model)
       
        
        # run float32 inference on calibration dataset to find the activations range
        for i, image in enumerate(images_files):
            #print('input size:',image.shape)
            if (i > self._max_img_num):
                break
            self.net_forward(self.model, image)
            print("loop stage 1 : %d" % (i))
            # find max threshold
            tensors = {}
            for name, feat in named_feats.items():
                print(name)
                tensors[name] = feat.flatten()
            collector.refresh_max_val(tensors)
            

        print(colored('max_vals', 'green'), collector.max_vals)
        distribution_intervals = collector.distribution_intervals
        for b_names in bottom_feat_names:
            assert len(b_names) > 1
            tmp_distribution_interval = 0
            b_is_eltwise = False
            for pre_feat_name in b_names:
                if (self.net_info[pre_feat_name]['type'] == 'Eltwise'):
                    b_is_eltwise = True
            if (b_is_eltwise):
                continue
            # distributions
            for pre_feat_name in b_names:
                tmp_distribution_interval = max(tmp_distribution_interval,
                                                distribution_intervals[pre_feat_name])
            for pre_feat_name in b_names:
                distribution_intervals[pre_feat_name] = tmp_distribution_interval

        # for each layer, collect histograms of activations
        print(colored("Collect histograms of activations:", 'cyan'))
        for i, image in enumerate(images_files):
            if (i > self._max_img_num):
                break
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
            b_is_eltwise = False
            for pre_feat_name in b_names:
                if (self.net_info[pre_feat_name]['type'] == 'Eltwise'):
                    b_is_eltwise = True
            if (b_is_eltwise):
                continue
            # distributions
            for pre_feat_name in b_names:
                tmp_distributions += distributions[pre_feat_name]
            for pre_feat_name in b_names:
                distributions[pre_feat_name] = tmp_distributions


        quantizer.quantize(distributions, distribution_intervals)
        bits = quantizer.bits

        # eltwise1 = eltwise0 + conv0 : bit(conv0)=bit(eltwise0)
        for b_names in bottom_feat_names:
            assert len(b_names) > 1
            elt_idx = 0
            b_is_eltwise = False
            for i,pre_feat_name in enumerate(b_names):
                if (self.net_info[pre_feat_name]['type'] == 'Eltwise'):
                    elt_idx = i
                    b_is_eltwise = True
            if (b_is_eltwise):      
                conv_id = 1 - elt_idx
                print(colored('bit conv:eltwise ', 'green'), bits[b_names[conv_id]],bits[b_names[elt_idx]])

                bits[b_names[conv_id]] = bits[b_names[elt_idx]]


        is_first_op = True
        for i, feat_name in enumerate(top_feat_names):
            if (feat_name == 'image'):
                feat_str = 'image' + ' ' + str(bits['image'])
            elif is_first_op:
                feat_str = self.cared_op_layer_names[i - 1] + ' ' + str(bits[feat_name])
                feat_str = feat_str + ' ' + str(bits['image'])
                is_first_op = False
            elif len(self.net_info[feat_name]['inputs']) > 0:
                feat_str = self.cared_op_layer_names[i-1] + ' ' + str(bits[feat_name])
                for inp_name in self.net_info[feat_name]['inputs']:
                    feat_str = feat_str + ' ' + str(bits[inp_name])
            else:
                raise NotImplementedError(self.net_info[feat_name])

            table_file_content.append(feat_str)

        with open(table_file, 'w') as f:
            for tabel_line in table_file_content:
                f.write(tabel_line + "\n")

    def regist_hook_outfeature(self, model):
        
        """
        创建hook获取每层的结果
        """
        out_feat = OrderedDict()
        hooks = []
        #print('layer num:', self.layers_num)

        all_op_type = self._all_op_type
        def _make_hook(m):
            
            def _hook(m, input, output):
                class_name = str(m.__class__).split(".")[-1].split("'")[0]
                
                layer_type = type(m).__name__
                idx = len(out_feat) % (int(self.layers_num + 1))
                if (idx == 0):
                    out_feat.clear()                  
                    out_feat['image'] = input[0].detach().cpu().numpy()
                    idx = len(out_feat)
                name_keys = "%s_%i" % (class_name, idx)
                out_feat[name_keys] = output.detach().cpu().numpy()

            if (type(m).__name__ in all_op_type):
                hooks.append(m.register_forward_hook(_hook))

        model.apply(_make_hook)


        # register hook
      
        return out_feat, hooks




    def init_dir(self):
        """
        ctreate the output folders if they don't exist
        """
        work_dir = self.config['OUTPUT']['WORK_DIR']
        weight_dir = self.config['OUTPUT']['WEIGHT_DIR']
        bias_dir = self.config['OUTPUT']['BIAS_DIR']
        final_weight_dir = self.config['OUTPUT']['FINAL_WEIGHT_DIR']
        final_bias_dir = self.config['OUTPUT']['FINAL_BIAS_DIR']

        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)
        if not os.path.exists(bias_dir):
            os.makedirs(bias_dir)
        if not os.path.exists(final_weight_dir):
            os.makedirs(final_weight_dir)
        if not os.path.exists(final_bias_dir):
            os.makedirs(final_bias_dir)

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

            param_np = param.detach().cpu().numpy()

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

        bits_co = {}
        # for weight quantity 1.dkl algorthm threashold for bit 2.max value for bit
        if(self._DKL_weight):
            collector.add_to_distributions(params)
            quantizer.quantize(collector.distributions, collector.distribution_intervals)
            bits_co = quantizer.bits
            print(colored('threshold:', 'green'), quantizer.threshold_value)
        else:
            for name in param_names:
                bit_int_d = math.ceil(math.log(collector.max_vals[name], 2))
                bit_bra_d_8 = int(8 - 1 - bit_int_d)
                bits_co[name] = bit_bra_d_8

        for name, bit in bits_co.items():
            param_quantity = np.around(params[name] * math.pow(2, bit))
            param_quantity = np.clip(param_quantity, -128, 127)
            #param_file_name = name.replace('.', '_')
            param_file_name = name
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



  
