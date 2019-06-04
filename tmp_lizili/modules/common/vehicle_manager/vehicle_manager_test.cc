#include "modules/common/vehicle_manager/vehicle_manager.h"

#include "gtest/gtest.h"

#include "modules/common/common_gflags.h"

namespace roadstar {
namespace common {

DEFINE_string(test_vehicle_data_path,
              "resources/vehicle_data/vehicle_manager_test_config.pb.txt",
              "The path of test vehicle info path");

class VehicleManagerTest : public ::testing::Test {
  void SetUp() override {
    FLAGS_vehicle_config_path = FLAGS_test_vehicle_data_path;
  }
};

TEST_F(VehicleManagerTest, GetVehicleLengthTest) {
  EXPECT_EQ(VehicleManager::GetVehicleLength(), 1.0);
}

TEST_F(VehicleManagerTest, GetVehicleWidthTest) {
  EXPECT_EQ(VehicleManager::GetVehicleWidth(), 2.0);
}

TEST_F(VehicleManagerTest, GetVehicleHeightTest) {
  EXPECT_EQ(VehicleManager::GetVehicleHeight(), 3.0);
}

TEST_F(VehicleManagerTest, GetVehicleInfo) {
  EXPECT_EQ(VehicleManager::GetVehicleInfo().length(), 1.0);
  EXPECT_EQ(VehicleManager::GetVehicleInfo().width(), 2.0);
  EXPECT_EQ(VehicleManager::GetVehicleInfo().height(), 3.0);
}

}  // namespace common
}  // namespace roadstar
