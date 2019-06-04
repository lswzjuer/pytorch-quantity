/******************************************************************************
 * Copyright 2018 The Roadstar Authors. All Rights Reserved.
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

#ifndef MODULES_COMMON_HDMAP_CLIENT_MOCK_HDMAP_INPUT_H_
#define MODULES_COMMON_HDMAP_CLIENT_MOCK_HDMAP_INPUT_H_

#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "modules/common/hdmap_client/hdmap_input.h"

namespace roadstar {
namespace common {

using ::testing::_;
using ::testing::Invoke;
using ::testing::Return;

class FakeHDMapInput : public HDMapInput {
 public:
  bool GetPointsOnRoad(const roadstar::common::PointENU &location,
                       double radius,
                       const std::vector<roadstar::common::PointENU> &points,
                       std::vector<int> *points_on_road) {
    for (int i = 0; i < static_cast<int>(points.size()); ++i) {
      const auto &point = points[i];
      if (std::hypot(point.x(), point.y()) <= 1 &&
          std::hypot(point.x() - location.x(), point.y() - location.y()) <=
              radius) {
        points_on_road->push_back(i);
      }
    }

    return true;
  }
};

// Mock a hdmap with a circle whose center is at (0, 0) and radius is 1.
class MockHDMapInput : public HDMapInput {
 public:
  // NOLINTNEXTLINE
  MOCK_METHOD4(GetPointsOnRoad,
               bool(const roadstar::common::PointENU &, const double,
                    const std::vector<roadstar::common::PointENU> &,
                    std::vector<int> *const));

  void DelegateToFake() {
    ON_CALL(*this, GetPointsOnRoad(_, _, _, _))
        .WillByDefault(Invoke(&fake_, &FakeHDMapInput::GetPointsOnRoad));
  }

  bool GetLocalLanemarkers(const roadstar::common::PointENU &location,
                           const double forward_distance,
                           const double backward_distance,
                           std::vector<Lanemarkers> *lanemarkers,
                           const int map_client_timeout_ms) override {
    Lanemarkers lanemarker;
    {
      hdmap::Lanemarker lane_mark;
      lane_mark.set_id(1);
      lane_mark.set_type(hdmap::LANEMARKER_SOLID_WHITE);
      auto curve = lane_mark.mutable_curve();
      curve->set_length(3);
      auto point = curve->add_points();
      point->set_x(0);
      point->set_y(1);
      point = curve->add_points();
      point->set_x(1);
      point->set_y(1);
      point = curve->add_points();
      point->set_x(2);
      point->set_y(1);
      point = curve->add_points();
      point->set_x(3);
      point->set_y(1);
      lanemarker.emplace_back(lane_mark);
    }
    {
      hdmap::Lanemarker lane_mark;
      lane_mark.set_id(2);
      lane_mark.set_type(hdmap::LANEMARKER_SOLID_WHITE);
      auto curve = lane_mark.mutable_curve();
      curve->set_length(1);
      auto point = curve->add_points();
      point->set_x(0);
      point->set_y(-1);
      point = curve->add_points();
      point->set_x(2);
      point->set_y(-1);
      point = curve->add_points();
      point->set_x(2);
      point->set_y(-1);
      point = curve->add_points();
      point->set_x(3);
      point->set_y(-1);
      lanemarker.emplace_back(lane_mark);
    }
    lanemarkers->emplace_back(lanemarker);
    return true;
  }

 private:
  FakeHDMapInput fake_;
};

}  // namespace common
}  // namespace roadstar

#endif  // MODULES_COMMON_HDMAP_CLIENT_MOCK_HDMAP_INPUT_H_
