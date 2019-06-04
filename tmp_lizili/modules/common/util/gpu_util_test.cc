#include "modules/common/util/gpu_util.h"
#include "gtest/gtest.h"

namespace roadstar {
namespace common {
namespace util {

TEST(GPUUtilTest, GPURemainTest) {
  std::vector<int> gpu_remain;
  std::string s =
      "Wed Oct 17 15:57:48 2018       \n"
      "+-----------------------------------------------------------------------"
      "------+ \n"
      "| NVIDIA-SMI 384.90                 Driver Version: 384.90              "
      "      | \n"
      "|-------------------------------+----------------------+----------------"
      "------+ \n"
      "| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile "
      "Uncorr. ECC | \n"
      "| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  "
      "Compute M. | \n"
      "|===============================+======================+================"
      "======| \n"
      "|   0  GeForce GTX 108...  Off  | 00000000:02:00.0 Off |                "
      "  N/A | \n"
      "| 29%   43C    P0    58W / 250W |      0MiB / 11172MiB |      0%      "
      "Default | \n"
      "+-------------------------------+----------------------+----------------"
      "------+ \n"
      "|   1  GeForce GTX 108...  Off  | 00000000:03:00.0 Off |                "
      "  N/A | \n"
      "| 29%   43C    P0    60W / 250W |      0MiB / 11172MiB |      0%      "
      "Default | \n"
      "+-------------------------------+----------------------+----------------"
      "------+ \n"
      "|   2  GeForce GTX 108...  Off  | 00000000:82:00.0 Off |                "
      "  N/A | \n"
      "| 29%   41C    P0    57W / 250W |      0MiB / 11172MiB |      0%      "
      "Default | \n"
      "+-------------------------------+----------------------+----------------"
      "------+ \n"
      "|   3  GeForce GTX 108...  Off  | 00000000:83:00.0 Off |                "
      "  N/A | \n"
      "| 29%   39C    P0    53W / 250W |      0MiB / 11172MiB |      0%      "
      "Default | \n"
      "+-------------------------------+----------------------+----------------"
      "------+ \n"
      "                                                                        "
      "        \n"
      "+-----------------------------------------------------------------------"
      "------+ \n"
      "| Processes:                                                       GPU "
      "Memory | \n"
      "|  GPU       PID   Type   Process name                             "
      "Usage      | \n"
      "|======================================================================="
      "======| \n"
      "|  No running processes found                                           "
      "      | \n"
      "+-----------------------------------------------------------------------"
      "------+ \n";
  GetGPURemain(s, &gpu_remain);
  EXPECT_GT(gpu_remain.size(), 0);
}

GTEST_API_ int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

}  // namespace util
}  // namespace common
}  // namespace roadstar
