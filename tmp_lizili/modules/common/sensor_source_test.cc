
#include "modules/common/sensor_source.h"

#include <iostream>
#include <unordered_map>

#include "gtest/gtest.h"

namespace roadstar {
namespace common {
namespace sensor {
TEST(SensorSourceTest, test) {
  EXPECT_TRUE(Is<Lidar>(LidarTailLeft));
  EXPECT_TRUE(!Is<Camera>(LidarTailLeft));
  EXPECT_TRUE(!Is<Radar>(LidarTailLeft));
  EXPECT_EQ(Name(LidarTailLeft), "Lidar(TailLeft)");

  // hash test
  std::unordered_map<SensorSource, int> map;
  map[LidarTailLeft] = 1;

  EXPECT_EQ(map[LidarTailLeft], 1);

  EXPECT_EQ(GetSensorType(UnknownSource), Unknown);
}
}  // namespace sensor
}  // namespace common
}  // namespace roadstar
