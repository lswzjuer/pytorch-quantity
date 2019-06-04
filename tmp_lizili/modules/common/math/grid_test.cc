
/******************************************************************************
 * Copyright 2017 The Roadstar Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/
#include "modules/common/math/grid.h"

#include <random>
#include "gtest/gtest.h"

namespace roadstar {
namespace common {
namespace math {
TEST(TestInit, Init) {
  Grid2d<float> grid({2, 2});
  grid(1, 0) = 1234;
  int grid_0_0 = grid(0, 0);
  int grid_1_0 = grid(1, 0);
  EXPECT_EQ(grid_0_0, 0);
  EXPECT_EQ(grid_1_0, 1234);
  Grid2d<float> grid3x3({3, 3});
  EXPECT_EQ(grid3x3.row(), 3);
  EXPECT_EQ(grid3x3.col(), 3);
}
TEST(TestIsPointInGrid, IsPointInGrid) {
  Grid2d<float> grid({3, 3}, {0.5, 0.6}, {0, 0});
  EXPECT_TRUE(grid.IsPointInGrid(1.2, 1.2));
  EXPECT_FALSE(grid.IsPointInGrid(1.2, 1.9));
  Grid2d<float> grid2({3, 3}, {0.5, 0.6}, {0.5, 0.6});
  std::array<int, 2> coord = grid2.GetGridCoord(1.2, 1.9);
  std::array<int, 2> coord2 = {static_cast<int>((1.2 - 0.5) / 0.5),
                               static_cast<int>((1.9 - 0.6) / 0.6)};
  EXPECT_EQ(coord, coord2);
}

TEST(TestGetOverlapArea, TestGetOverlapArea) {
  Grid2d<float> grid({3, 3});
  for (int i = 0; i < grid.row(); i++) {
    for (int j = 0; j < grid.col(); j++) {
      grid(i, j) = i * grid.col() + j + 1;
    }
  }
  AABox2d aabox1({0.5, 1.2}, {1.5, 1.6});
  AABox2d aabox2({0.8, 0.5}, {1.8, 2.5});
  AABox2d aabox3({1, 1}, {2.9, 2.9});
  std::vector<std::array<int, 2>> grid_coord1 = grid.GetOverlapArea(aabox1);
  std::vector<std::array<int, 2>> grid_coord2 = grid.GetOverlapArea(aabox2);
  std::vector<std::array<int, 2>> grid_coord3 = grid.GetOverlapArea(aabox3);
  std::vector<std::array<int, 2>> ans1 = {{0, 1}, {1, 1}};
  std::vector<std::array<int, 2>> ans2 = {{0, 0}, {0, 1}, {0, 2},
                                          {1, 0}, {1, 1}, {1, 2}};
  std::vector<std::array<int, 2>> ans3 = {{1, 1}, {1, 2}, {2, 1}, {2, 2}};
  EXPECT_EQ(grid_coord1, ans1);
  EXPECT_EQ(grid_coord2, ans2);
  EXPECT_EQ(grid_coord3, ans3);

  Grid2d<float> grid1({4, 4});
  const double pi = 3.1415;
  Box2d box2d(Vec2d(2, 2), pi / 4, std::sqrt(2) - 0.1, std::sqrt(2) - 0.1);
  std::vector<std::array<int, 2>> grid_coord = grid1.GetOverlapArea(box2d);
  std::vector<std::array<int, 2>> ans = {{1, 1}, {1, 2}, {2, 1}, {2, 2}};
  EXPECT_EQ(ans, grid_coord);

  Grid2d<float> grid2({5, 5});
  Box2d box2d_2(Vec2d(2.5, 2.5), pi / 4, 1.9 / std::sqrt(2),
                1.9 / std::sqrt(2));
  grid_coord = grid2.GetOverlapArea(box2d_2);
  ans = {{1, 2}, {2, 1}, {2, 2}, {2, 3}, {3, 2}};
}
TEST(TestGetOverlapAreaTime, TestGetOverlapAreaTime) {
  int iter_num = 10000;
  Grid2d<float> grid({100, 100});
  std::default_random_engine rand;
  const double pi = 3.1415926;
  std::uniform_real_distribution<double> theta(0.0, pi);
  std::uniform_real_distribution<double> length(1.0, 40.0);
  std::uniform_real_distribution<double> width(1.0, 40.0);
  std::uniform_real_distribution<double> center_x(1.0, 99.0);
  std::uniform_real_distribution<double> center_y(1.0, 99.0);
  for (int i = 0; i < iter_num; i++) {
    Box2d box2d(Vec2d(center_x(rand), center_y(rand)), theta(rand),
                length(rand), width(rand));
    auto coords = grid.GetOverlapArea(box2d);
  }
}

}  // namespace math
}  // namespace common
}  // namespace roadstar
