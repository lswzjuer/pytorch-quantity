#include "modules/common/geometry/lane_region.h"

#include <fstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "modules/common/geometry/curve.h"

namespace roadstar {
namespace common {
namespace geometry {

namespace {
constexpr int kSizeX = 1500;
constexpr int kSizeY = 1000;
const int kSize = kSizeY * kSizeX;
}  // namespace

bool LogMap(const std::vector<unsigned char> &map) {
  std::string file_name = "/roadstar/data/log/lane_region_test.txt";
  std::ofstream out(file_name.c_str());
  for (int i = 0; i < kSizeX; i++) {
    for (int j = 0; j < kSizeY; j++) {
      out << int(map.at(i * kSizeY + j)) << " ";
    }
    out << std::endl;
  }
  out.close();
  return true;
}

TEST(LaneRegionTest, StraightLaneTest) {
  std::vector<double> left_xs, left_ys, right_xs, right_ys;
  double slope = 0.1;
  double width = 3.5;
  for (int x = -100; x < 200; x += 1) {
    double center_y = slope * x;
    double left_y = center_y + width / 2;
    double right_y = center_y - width / 2;
    left_xs.push_back(x);
    right_xs.push_back(x);
    left_ys.push_back(left_y);
    right_ys.push_back(right_y);
  }
  Curve left_curve, right_curve;
  left_curve.FitCurve(left_xs, left_ys);
  right_curve.FitCurve(right_xs, right_ys);

  std::vector<unsigned char> map(kSize, 0);
  LaneRegion lane_region;
  lane_region.LabelLaneRegion(left_curve, right_curve, &map);
  EXPECT_TRUE(LogMap(map));
}

}  // namespace geometry
}  // namespace common
}  // namespace roadstar
