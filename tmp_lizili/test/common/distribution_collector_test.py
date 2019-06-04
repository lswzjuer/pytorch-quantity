import numpy as np

from roadtensor.common.quantity import DistributionCollector


class TestDistributionCollector(object):
    collector = DistributionCollector(['tensor'])

    def test_refresh_max_val(self):
        tensor = np.zeros((1, 3, 5, 5))
        tensor[..., 0] = 1
        tensor[..., 2] = -2

        tensors = {'tensor': tensor}
        self.collector.refresh_max_val(tensors)
        assert self.collector.max_vals['tensor'] == 2

        tensor[..., 0] = 0.5
        tensor[..., 2] = -1

        tensors = {'tensor': tensor}
        self.collector.refresh_max_val(tensors)
        assert self.collector.max_vals['tensor'] == 2

        tensor[..., 0] = 3
        tensor[..., 2] = -1

        tensors = {'tensor': tensor}
        self.collector.refresh_max_val(tensors)
        assert self.collector.max_vals['tensor'] == 3

    def test_distribution_intervals(self):
        assert self.collector.distribution_intervals['tensor'] - 1 * 3 / 2048 < 1e-6

    def test_add_to_distribution_worker(self):
        tensor = np.zeros((1, 3, 5, 5))
        tensor[0, 0, 0, :] = np.array([0.1, 1.2, 2.7, 2.9, 4.4])

        tensor_list, distributions = self.collector.add_to_distribution_worker(
            ['tensor'], {'tensor': tensor}, {'tensor': 1}, 5)
        assert tensor_list == ['tensor']
        assert (distributions[0] == np.array([1, 1, 2, 0, 1])).all(), distributions
