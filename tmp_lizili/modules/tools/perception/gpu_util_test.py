import unittest
import gpu_util

test_nvidia_output = '''Sat Oct 20 11:35:31 2018       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 384.90                 Driver Version: 384.90                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 108...  Off  | 00000000:02:00.0 Off |                  N/A |
| 29%   29C    P8     8W / 250W |   5073MiB / 11172MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   1  GeForce GTX 108...  Off  | 00000000:03:00.0 Off |                  N/A |
| 29%   33C    P8    16W / 250W |     10MiB / 11172MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   2  GeForce GTX 108...  Off  | 00000000:82:00.0 Off |                  N/A |
| 29%   28C    P8     8W / 250W |     10MiB / 11172MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   3  GeForce GTX 108...  Off  | 00000000:83:00.0 Off |                  N/A |
| 29%   28C    P8     8W / 250W |     10MiB / 11172MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      8671      C   ...dules/perception/lidar/lidar_perception  5063MiB |
+-----------------------------------------------------------------------------+
'''


class TestGetCommandOutPut(unittest.TestCase):
    def test_output(self):
        cmd = "ls / -d"
        result = gpu_util.GetCommandOutput(cmd)
        self.assertEqual(result, '/\n')


class TestGPUUtil(unittest.TestCase):
    def test_gpu_remain(self):
        result = gpu_util.GetGPURemain(test_nvidia_output)
        result_ground_true = [6099, 11162, 11162, 11162]
        self.assertEqual(result, result_ground_true)

    def test_assign_valid_gpu(self):
        gpu_id = gpu_util.AssignValidGPU(
            7000, 0, nvidia_output=test_nvidia_output)
        self.assertEqual(gpu_id, 1)


if __name__ == "__main__":
    unittest.main()
