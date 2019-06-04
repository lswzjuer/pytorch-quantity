import math
import numpy as np
import multiprocessing
from termcolor import colored


class DistributionCollector:

    def __init__(self,
                 tensor_list,
                 interval_num=2048,
                 statistic=1,
                 worker_num=1,
                 debug=False):
        """
        Used to collect the distribution of tensors from different layers.
        1. init this obj.
        2. put obj.refresh_max_val(tensor) in the loop of network forwarding.
        3. put obj.add_to_distributions(tensor) in the loop of network forwarding.
        4. use obj.distribution_intervals to get result.
        5. use obj.distributions to get result.

        Args:
            tensor_list: list, each value is a tensor name. For example, the tensor name
                should correspond to the param name when quantize the params of a net.
            interval_num: int, we will devide 0 ~ max-val into interval_num bins to get
                the discrete distribution of the tensor.
            statistic: int.
            worker_num: int.
            debug: bool.
        """
        self._tensor_list = tensor_list
        self._interval_num = interval_num
        self._statistic = statistic
        self._worker_num = worker_num
        self._debug = debug

        self._distributions = {}
        self._max_vals = {}
        for tensor_name in self._tensor_list:
            self._distributions[tensor_name] = np.zeros((self._interval_num,), dtype=np.int32)
            self._max_vals[tensor_name] = 0

        self._max_vals_refreshed_flag = False
        self._added_to_distributions_flag = False

    @property
    def max_vals(self):
        assert self._max_vals_refreshed_flag, "Please use refresh_max_val() first."
        return self._max_vals

    @property
    def distribution_intervals(self):
        """We will devide 0 - max_val into interval_num bins. This function will return the
        width of each bin of each layer.
        """
        assert self._max_vals_refreshed_flag, "Please use refresh_max_val() first."
        distribution_intervals = {}
        for tensor_name in self._tensor_list:
            distribution_intervals[tensor_name] = \
                self._statistic * self._max_vals[tensor_name] / self._interval_num + 1e-12
        self._distribution_intervals = distribution_intervals
        return distribution_intervals

    @property
    def distributions(self):
        assert self._added_to_distributions_flag, "Please use add_to_distributions() first."
        return self._distributions

    def refresh_max_val(self, tensors):
        """Put this function in the loop of the network forwarding to refresh
        the max abs val of each tensor.
        """
        self._max_vals_refreshed_flag = True
        for tensor_name in self._tensor_list:
            tensor = tensors[tensor_name]
            max_val = max(abs(np.max(tensor)), abs(np.min(tensor)))
            self._max_vals[tensor_name] = max(self._max_vals[tensor_name], max_val)

    def add_to_distributions(self, tensors):
        """Put this function in the loop of the network forwarding to refresh
        the distribution of each tensor.
        """
        if self._debug and self._added_to_distributions_flag:
            return

        self._added_to_distributions_flag = True
        if not hasattr(self, '_distribution_intervals'):
            print(colored("interval:", 'green'), self.distribution_intervals)

        pool = multiprocessing.Pool(processes=self._worker_num)
        amount_per_worker = int(math.floor(len(self._tensor_list) / self._worker_num))
        results = []
        for worker_i in range(self._worker_num):
            sub_tensor_list = self._tensor_list[worker_i * amount_per_worker:
                                                (worker_i + 1) * amount_per_worker]
            if worker_i == 0:
                sub_tensor_list += self._tensor_list[self._worker_num * amount_per_worker:]
            sub_tensors, sub_distribution_intervals = {}, {}
            for tensor_name in sub_tensor_list:
                sub_tensors[tensor_name] = tensors[tensor_name]
                sub_distribution_intervals[tensor_name] = self._distribution_intervals[tensor_name]
            result = pool.apply_async(
                run,
                args=(DistributionCollector,
                      sub_tensor_list,
                      sub_tensors,
                      sub_distribution_intervals,
                      self._interval_num,
                      self._debug))
            results.append(result)
        pool.close()
        pool.join()

        for result in results:
            tensor_list, sub_distributions = result.get()
            for (tensor_name, distribution) in zip(tensor_list, sub_distributions):
                self._distributions[tensor_name] += distribution
        pool.terminate()

    @staticmethod
    def add_to_distribution_worker(tensor_list, tensors, intervals, interval_num, debug=False):
        if debug:
            return tensor_list, [np.ones(interval_num, dtype=np.int32)
                                 for _ in range(len(tensor_list))]

        def _add_to_distribution(data, interv_num, interval):
            distribution = [0 for _ in range(interv_num)]
            max_index = interv_num - 1
            # each bin includes the values in (x, y], so we ignore 0 in the data.
            indexes = np.minimum((abs(data[data != 0]) / interval).astype(np.int32), max_index)
            # Note that distribution[indexes] += 1 is not work.
            for index in indexes:
                distribution[index] += 1
            return np.array(distribution, dtype=np.int32)

        distributions = []
        for tensor_name in tensor_list:
            distribution = _add_to_distribution(
                tensors[tensor_name], interval_num, intervals[tensor_name])
            distributions.append(distribution)
        return tensor_list, distributions


def run(cls_instance, *args):
    """Compatible with Python2."""
    return cls_instance.add_to_distribution_worker(*args)
