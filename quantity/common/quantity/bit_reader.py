from termcolor import colored
from collections import OrderedDict, defaultdict, Counter


class BitReader:

    def __init__(self,
                 feat_table=None,
                 weight_table=None):
        """
        Used to read the table file we saved in weight-quantizer and activation-quantizer.

        Args:
            feat_table: file path.
            weight_table: file path.
        """
        self._feat_table = feat_table
        self._weight_table = weight_table

    def get_feat_info(self):
        assert self._feat_table

        feat_bits, infeat_bits = {}, {}
        count_feat = 0
        with open(self._feat_table, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            layer_name, bit = line.split(' ')[:2]
            feat_bits[layer_name] = int(eval(bit))
            infeat_bits[layer_name] = line.split(' ')[2:]
            count_feat += 1
        print(colored("feat count:", 'green'), count_feat)
        return feat_bits, infeat_bits

    def get_weight_info(self):
        assert self._weight_table

        weight_bits, bias_bits = OrderedDict(), OrderedDict()
        count_weight, count_bias = 0, 0

        with open(self._weight_table, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            contents = line.split(' ')
            layer_name, bit = contents
            bit = int(eval(bit))

            if layer_name.endswith('.weight'):
                layer_name = layer_name[:-7]
                count_weight += 1
                weight_bits[layer_name] = bit
            elif layer_name.endswith('.bias'):
                layer_name = layer_name[:-5]
                count_bias += 1
                bias_bits[layer_name] = bit
            else:
                print(colored("Unknow layer name {}".format(layer_name), 'red'))

        print(colored("weight count:", 'green'), count_weight)
        print(colored("bias count:", 'green'), count_bias)
        return weight_bits, bias_bits
