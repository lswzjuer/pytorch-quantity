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

#ifndef MODULES_COMMON_HDMAP_CLIENT_HDMAP_INPUT_H_
#define MODULES_COMMON_HDMAP_CLIENT_HDMAP_INPUT_H_

#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "modules/common/hdmap_client/map_client.h"
#include "modules/common/macro.h"
#include "modules/common/status/status.h"
#include "modules/msgs/hdmap/proto/hdmap_common.pb.h"
#include "modules/msgs/hdmap/proto/lanemarker.pb.h"
#include "modules/msgs/hdmap/proto/traffic_light.pb.h"
#include "modules/msgs/localization/proto/localization.pb.h"

namespace roadstar {
namespace common {

class HDMapInput {
 public:
  /* @brief Get cached polygons used for filtering
   * @return Polygons
   */
  virtual bool GetLocalPolygons(
      const roadstar::common::PointENU &location, const double radius,
      std::vector<roadstar::common::Polygon> *const polygons);

  virtual bool GetPointsOnRoad(
      const roadstar::common::PointENU &location, const double radius,
      const std::vector<roadstar::common::PointENU> &points,
      std::vector<int> *const points_on_road);

  virtual bool GetLocalPathElements(
      const roadstar::common::PointENU &location, const double forward_distance,
      const double backward_distance,
      roadstar::hdmap::MapElements *const map_elements);

  virtual bool GetLocalMapElements(
      const roadstar::common::PointENU &location, const double radius,
      roadstar::hdmap::MapElements *const map_elements);

  virtual bool GetPointsInfo(
      const roadstar::common::PointENU &location, const double radius,
      const std::vector<roadstar::common::PointENU> &points,
      std::vector<roadstar::hdmap::PointInfo> *const points_info);

  virtual bool GetLocalLanemarkers(const roadstar::common::PointENU &location,
                                   const double forward_distance,
                                   const double backward_distance,
                                   std::vector<Lanemarkers> *lanemarkers,
                                   const int map_client_timeout_ms);

  virtual bool GetLocalLanemarkers(const roadstar::common::PointENU &location,
                                   const double forward_distance,
                                   const double backward_distance,
                                   std::vector<Lanemarkers> *lanemarkers) {
    return GetLocalLanemarkers(location, forward_distance, backward_distance,
                               lanemarkers, -1);
  }

  virtual bool GetLocalMergedLanemarkers(
      const roadstar::common::PointENU &location, const double forward_distance,
      const double backward_distance,
      std::vector<hdmap::MergedLanemarker> *lanemarkers,
      const int map_client_timeout_ms);

  virtual bool GetLocalMergedLanemarkers(
      const roadstar::common::PointENU &location, const double forward_distance,
      const double backward_distance,
      std::vector<hdmap::MergedLanemarker> *lanemarkers) {
    return GetLocalMergedLanemarkers(location, forward_distance,
                                     backward_distance, lanemarkers, -1);
  }

  virtual bool GetLocalTrafficLight(
      const roadstar::common::PointENU &location, const double forward_distance,
      roadstar::hdmap::TrafficLight *const traffic_light);

  virtual bool GetCurrMap(std::string *map_name);

  bool GetCurrMapRoute(std::string *map_route_name);

 private:
  MapClient map_client_;

  friend class MockHDMapInput;
  friend class FakeHDMapInput;

  static void MergeLanemarker(
      hdmap::MergedLanemarker *prev_merged_lanemarker,
      hdmap::MergedLanemarker *next_merged_lanemarker,
      std::unordered_map<int, hdmap::MergedLanemarker *> *merged_lanemarkers);

  static void MergeLanemarker(
      hdmap::MergedLanemarker *merged_lanemarker, hdmap::Lanemarker *lanemarker,
      std::unordered_map<int, hdmap::MergedLanemarker *> *merged_lanemarkers,
      bool back);

  DECLARE_SINGLETON(HDMapInput);
};

}  // namespace common
}  // namespace roadstar

#endif  // MODULES_COMMON_HDMAP_CLIENT_HDMAP_INPUT_H_
