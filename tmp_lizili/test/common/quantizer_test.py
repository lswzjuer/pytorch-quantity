import numpy as np

from roadtensor.common.quantity import Quantizer


class TestQuantizer(object):
    quantizer = Quantizer(['tensor'])

    def test_quantize_worker(self):
        tensor = np.zeros((1, 3, 5, 5))
        tensor[0, 0, 0, :] = np.array([1, 2, 3, 4, 5])

        tensors = {'tensor': tensor}
        tensor_list, bits = self.quantizer.quantize_worker(
            ['tensor'], {'tensor': np.array([1, 1, 1, 1, 1])}, {'tensor': 1})
        assert tensor_list == ['tensor']
        assert bits[0] == 4
