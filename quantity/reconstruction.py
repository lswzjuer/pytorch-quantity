# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2019-06-17 10:43:49
# @Last Modified by:   liusongwei
# @Last Modified time: 2019-06-21 18:25:38
# -*- coding:utf-8 -*i
import os
import yaml
from termcolor import colored
from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision.models as visionmodels
from common.quantity import walk_dirs, merge_bn
from common.quantity import BitReader,Eltwise,Concat,Identity,NewConv2d,NewAdd,NewLinear,TestConv,TestLinear
from common.quantity import merge_bn

def Run_model_quantizer(model,quantity_cfg):
    '''
    :param model:  the model must be one which have been merged BN layer
                   NOTE: the model must be eval() state, it means model=model.eval()
    :param quantity_cfg:  the loaded common config in configs.yaml
    :return: None    the quantity information have been saved in some files

    '''
    from weight_quantizer import init_dir ,weight_quantize
    from activation_quantizer import build_net_structure, activation_quantize
    from rewriter import BiasReWriter

    # First of all quantify the weight
    init_dir(quantity_cfg)
    weight_quantize(model,quantity_cfg)
    print(colored("Weight quantization has been done","green"))


    # The activation value is then quantified
    max_cali_num = quantity_cfg['SETTINGS']['MAX_CALI_IMG_NUM']
    cali_data_dir = quantity_cfg['PATH']['DATA_PATH']
    cali_file_type=quantity_cfg['SETTINGS']['CALI_FILE_TYPE']
    # load cali samples
    file_names = walk_dirs(cali_data_dir, file_type=cali_file_type)[:max_cali_num]
    net_info,old_info = build_net_structure(model)
    activation_quantize(model, net_info, file_names, quantity_cfg)
    print(colored("Activation quantization has been done","green"))


    # We have got the quantized information, but need to modify bias_frac bit = output_frac bit
    weight_dir = quantity_cfg['OUTPUT']['WEIGHT_DIR']
    bias_dir = quantity_cfg['OUTPUT']['BIAS_DIR']
    output_weight_dir = quantity_cfg['OUTPUT']['FINAL_WEIGHT_DIR']
    output_bias_dir = quantity_cfg['OUTPUT']['FINAL_BIAS_DIR']
    weight_table = quantity_cfg['OUTPUT']['WEIGHT_BIT_TABLE']
    feat_table = quantity_cfg['OUTPUT']['FEAT_BIT_TABLE']
    max_shift_limit = quantity_cfg['SETTINGS']['MAX_SHIFT']

    if not os.path.exists(output_weight_dir):
        os.makedirs(output_weight_dir)

    if not os.path.exists(output_bias_dir):
        os.makedirs(output_bias_dir)

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
    new_aligned_bias = rewriter.bias_output_bit_align(bias_bits, feat_bits)
    # now, bias_bits == feat_bits == new_bias.
    rewriter.rewrite_bias_table(bias_bits, new_aligned_bias)
    rewriter.rewrite_bias_dir(bias_bits, new_aligned_bias)

    print(colored("Add max shift limitation:", 'cyan'))
    new_weight = rewriter.max_shift_limit_weight(feat_bits, infeat_bits, weight_bits)
    if not (new_weight is True):
        # now, weight_bits <= max_shift + output_bits - input_bits.
        rewriter.rewrite_weight_table(weight_bits, new_weight)
        rewriter.rewrite_weight_dir(weight_bits, new_weight)
    print(colored("Rewrite quantization information has been done","green"))


# Rebuild model class
class Reconstruction(object):
    """docstring for Reconstruction"""

    def __init__(self, model):
        super(Reconstruction, self).__init__()
        self.model = model
        self.load_configs()

    def load_configs(self):

        #read quantity config file
        with open("./configs.yml") as f:
            self.config = yaml.load(f)

    def get_quantity_information(self):

        # the cared_op_type list, it must be same as the list used in the quantification process
        cared_op_type=self.config['SETTINGS']['CARE_OP_TYPE']
        # Read quantization information we need
        weight_file=self.config['OUTPUT']['WEIGHT_BIT_TABLE']
        feat_file=self.config['OUTPUT']['FEAT_BIT_TABLE']

        bit_reader=BitReader(feat_table=feat_file, weight_table=weight_file)
        # weight table and feat table
        weight_bits, bias_bits = bit_reader.get_weight_info()
        feat_bits, infeat_bits = bit_reader.get_feat_info()
        # 如果某个带参数的层比如 deconv 没有加入activation_quantizer里面的 cared_op_type
        # 那么就会出现层名在weight_bits里面，但是不再feat_bits里面，就会出错
        all_quantize_infor=OrderedDict()
        for name ,bit in weight_bits.items():
            # the layer whcih have weight and bias must be also in feat table
            assert name in feat_bits, "{} not in {}".format(name, feat_bits)
            assert name in infeat_bits, "{} not in {}".format(name, infeat_bits)
            # Restore the original layer name
            new_name=name
            weight_bit=bit
            bias_bit=bias_bits[name]
            output_bit=feat_bits[name]
            input_bit=int(infeat_bits[name][0])
            print("name: {} weight:{} bias:{} in:{} out:{}".format(new_name,weight_bit,bias_bit,input_bit,output_bit))
            # Verification bit  accuracy
            # assert bias_bit==output_bit ,"the bias bit != output bit in layer :{}".format(new_name)
            if new_name not in all_quantize_infor:
                new_dict={}
                new_dict['weight_bit']=weight_bit
                new_dict['bias_bit'] = output_bit
                new_dict['output_bit'] = output_bit
                new_dict['input_bit'] = input_bit
                all_quantize_infor[new_name]=new_dict


        # some layer without parameters in feat table but not in weight table,
        # we should collete them
        for name, bit in feat_bits.items():
            new_name=name
            if new_name in all_quantize_infor:
                continue
            else:
                new_dict={}
                new_dict['weight_bit']=None
                new_dict['bias_bit'] = None
                new_dict['output_bit'] = bit
                if new_name=='image':
                    new_dict['input_bit']=None
                else:
                    #assert int(infeat_bits[name][0])==bit, "inputbit!=outputbit in {} ".format(new_name)
                    new_dict['input_bit'] = int(infeat_bits[name][0])
                all_quantize_infor[new_name]=new_dict

        # The convolution layer and full connection layer are needed to reconstruct the network
        quantize_infor_keys=all_quantize_infor.keys()
        for name, module in self.model.named_modules():
            module_type=type(module).__name__
            print("name is : {}  module_type is : {}".format(name,module_type))
            if name in quantize_infor_keys and module_type in cared_op_type :
                #The module already contains information with a parameter layer
                all_quantize_infor[name]['layer'] = module
                all_quantize_infor[name]['layer_type']=module_type

        return all_quantize_infor


    def ReconModel(self,all_quantize_infor,new_model_path):
        '''
        :param all_quantize_infor:
        :return: the new quantity model
        '''

        for name, module in self.model.named_modules():
            module_type=type(module).__name__
            # replace conv layer
            if module_type=="Conv2d":
                assert all_quantize_infor[name]['layer_type']=="Conv2d", "layer type wrong"
                find_module = self.model
                name_split = name.split('.')
                if len(name_split) == 1:
                    find_module.add_module(name_split[0],
                                           NewConv2d(module,all_quantize_infor[name]))
                    print(colored('The layer change: {} ==>NewConv2d '.format(name), 'green'))
                elif len(name_split) >= 2:
                    father_list = name_split[:-1]
                    for n in father_list:
                        find_module = getattr(find_module, n)
                    find_module.add_module(name_split[-1],
                                           NewConv2d(module,all_quantize_infor[name]))
                    print(colored('The layer change: {} ==>NewConv2d '.format(name), 'green'))
                else:
                    raise ValueError("the conv layer name is wrong")
            # replace linear layer
            elif module_type=="Linear":
                assert all_quantize_infor[name]['layer_type'] == "Linear", "layer type wrong"
                find_module = self.model
                name_split = name.split('.')
                if len(name_split) == 1:
                    find_module.add_module(name_split[0],
                                           NewLinear(module,all_quantize_infor[name]))
                    print(colored('The layer change: {} ==>NewLinear '.format(name), 'green'))
                elif len(name_split) >= 2:
                    father_list = name_split[:-1]
                    for n in father_list:
                        find_module = getattr(find_module, n)
                    find_module.add_module(name_split[-1],
                                           NewLinear(module,all_quantize_infor[name]))
                    print(colored('The layer change: {} ==>NewLinear '.format(name), 'green'))
                else:
                    raise ValueError("the linear layer name is wrong")
            # replace eltwise layer
            elif module_type=="Eltwise":
                assert all_quantize_infor[name]['layer_type']=="Eltwise", "layer type wrong"
                find_module = self.model
                name_split = name.split('.')
                if len(name_split) == 1:
                    find_module.add_module(name_split[0],
                                           NewAdd())
                    print(colored('The layer change: {} ==>NewAdd '.format(name), 'green'))
                elif len(name_split) >= 2:
                    father_list = name_split[:-1]
                    for n in father_list:
                        find_module = getattr(find_module, n)
                    find_module.add_module(name_split[-1],
                                           NewAdd())
                    print(colored('The layer change: {} ==>NewAdd '.format(name), 'green'))
                else:
                    raise ValueError("the Eltwise layer name is wrong")
            else:
                #raise  NotImplementedError(colored("Please add new layer change support !!!!"),'red')
                continue
        print(colored("Model reconstruction successfully !",'green'))
        torch.save(self.model,new_model_path)
        return self.model

    def ReconTest(self,all_quantize_infor,new_model_path):

        '''
        w->qw->dqw
        bias->qbias->dqbias
        output->qoutput->dqoutput
        '''
        for name, module in self.model.named_modules():
            module_type = type(module).__name__
            # replace conv layer
            if module_type == "Conv2d":
                assert all_quantize_infor[name]['layer_type'] == "Conv2d", "layer type wrong"
                find_module = self.model
                name_split = name.split('.')
                if len(name_split) == 1:
                    find_module.add_module(name_split[0],
                                           TestConv(name,module, all_quantize_infor[name],new_model_path))

                    print("name is : {}\n quantize_info is : {}".format(name,all_quantize_infor[name]))
                    print(colored('The layer change: {} ==>NewConv2d '.format(name), 'green'))
                elif len(name_split) >= 2:
                    father_list = name_split[:-1]
                    for n in father_list:
                        find_module = getattr(find_module, n)
                    find_module.add_module(name_split[-1],
                                           TestConv(name, module, all_quantize_infor[name],new_model_path))

                    print("name is : {}\n quantize_info is : {}".format(name,all_quantize_infor[name]))
                    print(colored('The layer change: {} ==>NewConv2d '.format(name), 'green'))
                else:
                    raise ValueError("the conv layer name is wrong")
            # replace linear layer
            elif module_type == "Linear":
                assert all_quantize_infor[name]['layer_type'] == "Linear", "layer type wrong"
                find_module = self.model
                name_split = name.split('.')
                if len(name_split) == 1:
                    find_module.add_module(name_split[0],
                                           TestLinear(name, module, all_quantize_infor[name],new_model_path))

                    print("name is : {}\n quantize_info is : {}".format(name,all_quantize_infor[name]))
                    print(colored('The layer change: {} ==>NewLinear '.format(name), 'green'))
                elif len(name_split) >= 2:
                    father_list = name_split[:-1]
                    for n in father_list:
                        find_module = getattr(find_module, n)
                    find_module.add_module(name_split[-1],
                                           TestLinear(name, module, all_quantize_infor[name],new_model_path))

                    print("name is : {}]\n quantize_info is : {}".format(name,all_quantize_infor[name]))
                    print(colored('The layer change: {} ==>NewLinear '.format(name), 'green'))
                else:
                    raise ValueError("the linear layer name is wrong")

            # replace eltwise layer
            elif module_type == "Eltwise":
                assert all_quantize_infor[name]['layer_type'] == "Eltwise", "layer type wrong"
                find_module = self.model
                name_split = name.split('.')
                if len(name_split) == 1:
                    find_module.add_module(name_split[0],
                                           NewAdd())
                    print("name is : {}\n quantize_info is : {}".format(name,all_quantize_infor[name]))
                    print(colored('The layer change: {} ==>NewAdd '.format(name), 'green'))
                elif len(name_split) >= 2:
                    father_list = name_split[:-1]
                    for n in father_list:
                        find_module = getattr(find_module, n)
                    find_module.add_module(name_split[-1],
                                           NewAdd())

                    print("name is : {}\n quantize_info is : {}".format(name,all_quantize_infor[name]))
                    print(colored('The layer change: {} ==>NewAdd '.format(name), 'green'))
                else:
                    raise ValueError("the Eltwise layer name is wrong")
            else:
                # raise  NotImplementedError(colored("Please add new layer change support !!!!"),'red')
                continue

        print(colored("Model reconstruction successfully !", 'green'))
        torch.save(self.model, new_model_path)
        return self.model
        
    def merge_bn(self):
        '''
        input: the origin model
        output: the model without bn layer
        '''
        # self.model=merge_bn(self.model) 
        return merge_bn(self.model)

