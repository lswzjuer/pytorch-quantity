import argparse
import os
import os.path as osp
import yaml
import json
import numpy as np
from termcolor import colored

from roadtensor.common.quantity import BitReader, walk_dirs


class BiasReWriter:

    def __init__(self,
                 weight_dir,
                 bias_dir,
                 output_weight_dir,
                 output_bias_dir,
                 weight_file,
                 feat_file,
                 max_shift_limit=None):
        """Add the bit info to the prototxt file. The bias and output bit
        is aligned.
        """
        self._weight_dir = weight_dir
        self._bias_dir = bias_dir
        self._output_weight_dir = output_weight_dir
        self._output_bias_dir = output_bias_dir
        self._weight_file = weight_file
        self._max_shift_limit = max_shift_limit
        self._bit_reader = BitReader(feat_table=feat_file, weight_table=weight_file)

    def get_weight_info(self):
        return self._bit_reader.get_weight_info()

    def get_feat_info(self):
        return self._bit_reader.get_feat_info()

    def bias_output_bit_align(self, bias_bits, feat_bits):
        # assume the feat bit is larger than bias bit.
        new_aligned_bits = {}
        for key in bias_bits.keys():
            bias_bit, feat_bit = bias_bits[key], feat_bits[key]
            #new_aligned_bits[key] = min(bias_bit, feat_bit)
            new_aligned_bits[key] = min(bias_bit, feat_bit)

            print(colored("{}: ".format(key), 'green'), "{}, {} => {}".format(
                bias_bit, feat_bit, new_aligned_bits[key]))
        return new_aligned_bits

    def rewrite_bias_dir(self, old_bias_bits, new_bias_bits):
        files_path = walk_dirs(self._bias_dir, file_type='.json')
        for file_path in files_path:
            assert file_path.endswith('_bias.json'), file_path
            layer_name = osp.basename(file_path)[:-10]

            if not layer_name in new_bias_bits.keys():
                print(colored("Can't find {} in weight table, but json file exists.".format(
                    layer_name), 'red'))
                continue

            assert layer_name in new_bias_bits.keys(), layer_name
            with open(file_path, 'r') as f:
                lines = json.load(f)

            lines = np.array(lines, dtype=np.float32)
            lines = lines / 2 ** old_bias_bits[layer_name] * 2 ** new_bias_bits[layer_name]
            lines = np.around(lines).astype(np.int8).tolist()

            output_path = osp.join(self._output_bias_dir, osp.basename(file_path))
            with open(output_path, 'w') as f:
                json.dump(lines, f, indent=4)

    def rewrite_bias_table(self, old_bias_bits, new_bias_bits):
        with open(self._weight_file, 'r') as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            line = line.strip()
            name, bit = line.split(' ')[:2]
            if name[:-5] in old_bias_bits.keys():
                bit = str(new_bias_bits[name[:-5]])
            new_lines.append("{} {}\n".format(name, bit))

        with open(self._weight_file, 'w') as f:
            f.writelines(new_lines)

    def max_shift_limit_weight(self, feat_bits, infeat_bits, weight_bits):
        if self._max_shift_limit is None:
            return True

        new_weight_bits = {}
        for name, bit in weight_bits.items():
            assert name in feat_bits, "{} not in {}".format(name, feat_bits)
            assert name in infeat_bits, "{} not in {}".format(name, infeat_bits)
            new_weight_bit = weight_bits[name]

            assert len(set(infeat_bits[name])) == 1, infeat_bits[name]
            input_bit = int(infeat_bits[name][0])
            output_bit = feat_bits[name]

            need_rewrite = False
            if weight_bits[name] + input_bit - output_bit > self._max_shift_limit:
                new_weight_bit = self._max_shift_limit - infeat_bits[name] + feat_bits[name]
                print(colored("weight bit: {} => {}".format(weight_bits[name], new_weight_bit),
                              'green'))
                need_rewrite = True
            new_weight_bits[name] = new_weight_bit

        if not need_rewrite:
            print(colored("Nothing needs to change.", 'green'))
            return True

        return new_weight_bits

    def rewrite_weight_dir(self, old_weight_bits, new_weight_bits):
        files_path = walk_dirs(self._weight_dir, file_type='.json')
        for file_path in files_path:
            assert file_path.endswith('_weight.json'), file_path
            layer_name = osp.basename(file_path)[:-12]

            if not layer_name in new_weight_bits.keys():
                print(colored("Can't find {} in weight table, but json file exists.".format(
                    layer_name), 'red'))
                continue

            with open(file_path, 'r') as f:
                lines = json.load(f)
            lines = np.array(lines, dtype=np.float32)
            lines = lines / 2 ** old_weight_bits[layer_name] * 2 ** new_weight_bits[layer_name]

            lines = np.around(lines).astype(np.int8).tolist()

            output_path = osp.join(self._output_weight_dir, osp.basename(file_path))
            with open(output_path, 'w') as f:
                json.dump(lines, f, indent=4)

    def rewrite_weight_table(self, old_weight_bits, new_weight_bits):
        with open(self._weight_file, 'r') as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            line = line.strip()
            name, bit = line.split(' ')[:2]
            if name[:-7] in old_weight_bits.keys():
                bit = str(new_weight_bits[name[:-7]])
            new_lines.append("{} {}\n".format(name, bit))

        with open(self._weight_file, 'w') as f:
            f.writelines(new_lines)


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--yml', dest='yml',
                        default='roadtensor/components/tools/quantity/configs.yml',
                        help="path to the yml file. the args below are NOT"
                             "needed if yml file is provided.")
    args = parser.parse_args()

    assert os.path.isfile(args.yml), args.yml
    with open(args.yml) as f:
        configs = yaml.load(f)

    weight_dir = configs['OUTPUT']['WEIGHT_DIR']
    bias_dir = configs['OUTPUT']['BIAS_DIR']
    output_weight_dir = configs['OUTPUT']['FINAL_WEIGHT_DIR']
    output_bias_dir = configs['OUTPUT']['FINAL_BIAS_DIR']
    weight_table = configs['OUTPUT']['WEIGHT_BIT_TABLE']
    feat_table = configs['OUTPUT']['FEAT_BIT_TABLE']
    max_shift_limit = configs['SETTINGS']['MAX_SHIFT']

    if not osp.exists(output_weight_dir):
        os.makedirs(output_weight_dir)

    if not osp.exists(output_bias_dir):
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
    #new_aligned_bias = rewriter.bias_output_bit_align(bias_bits, feat_bits)
    # now, bias_bits == feat_bits == new_bias.
    rewriter.rewrite_bias_table(bias_bits, feat_bits)
    rewriter.rewrite_bias_dir(bias_bits, feat_bits)

    print(colored("Add max shift limitation:", 'cyan'))
    new_weight = rewriter.max_shift_limit_weight(feat_bits, infeat_bits, weight_bits)
    if new_weight:
        # now, weight_bits <= max_shift + output_bits - input_bits.
        rewriter.rewrite_weight_table(weight_bits, new_weight)
        rewriter.rewrite_weight_dir(weight_bits, new_weight)

    print("Done!")


if __name__ == '__main__':
    main()
