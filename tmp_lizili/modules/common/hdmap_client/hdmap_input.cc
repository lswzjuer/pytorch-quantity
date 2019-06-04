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

#include "modules/common/hdmap_client/hdmap_input.h"

#include <limits>
#include <unordered_set>
#include <utility>

#include "modules/common/adapters/adapter_manager.h"
#include "modules/common/common_gflags.h"
#include "modules/common/geometry/curve.h"
#include "modules/common/log.h"
#include "modules/common/math/polygon2d.h"

using roadstar::common::PointENU;
using roadstar::hdmap::MapUnit;
using roadstar::hdmap::PointInfo;

namespace roadstar {
namespace common {

namespace {

constexpr int kMapClientTimeoutMs = 10;

}  // namespace

HDMapInput::HDMapInput() {}

bool HDMapInput::GetLocalPolygons(
    const PointENU &location, const double radius,
    std::vector<roadstar::common::Polygon> *const polygons) {
  // For localization offset
  double real_location_x = location.x() + location.offset_x();
  double real_location_y = location.y() + location.offset_y();

  auto status = map_client_.GetLocalBoundary(
      real_location_x, real_location_y, radius, polygons, kMapClientTimeoutMs);
  if (!status.ok()) {
    AERROR_EVERY(100) << "Failed to call map service GetLocalBoundary";
    return false;
  }

  // For localization offset
  for (auto &polygon : *polygons) {
    for (int i = 0; i < polygon.points_size(); i++) {
      auto mutable_point = polygon.mutable_points(i);
      mutable_point->set_x(mutable_point->x() - location.offset_x());
      mutable_point->set_y(mutable_point->y() - location.offset_y());
    }
  }

  return true;
}

bool HDMapInput::GetPointsOnRoad(const PointENU &location, const double radius,
                                 const std::vector<PointENU> &points,
                                 std::vector<int> *const points_on_road) {
  std::vector<PointInfo> points_info;

  if (!GetPointsInfo(location, radius, points, &points_info)) {
    return false;
  }

  for (size_t i = 0; i < points_info.size(); ++i) {
    if (points_info[i].map_unit().type() != MapUnit::MAP_UNIT_NONE) {
      points_on_road->push_back(i);
    }
  }

  return true;
}

bool HDMapInput::GetLocalPathElements(
    const PointENU &location, const double forward_distance,
    const double backward_distance,
    roadstar::hdmap::MapElements *const map_elements) {
  roadstar::hdmap::Path path;
  auto status =
      map_client_.GetLocalPath(location.x(), location.y(), forward_distance,
                               backward_distance, &path, kMapClientTimeoutMs);
  if (!status.ok()) {
    AERROR_EVERY(100) << "Failed to call map service GetLocalPath";
    return false;
  }
  std::vector<roadstar::hdmap::MapUnit> map_units;
  for (const auto &unit : path.path_units()) {
    map_units.push_back(unit);
  }
  status = map_client_.RetrieveMapElements(map_units, map_elements,
                                           kMapClientTimeoutMs);
  if (!status.ok()) {
    AERROR_EVERY(100) << "Failed to call map service RetrieveMapElements";
    return false;
  }
  return true;
}

bool HDMapInput::GetPointsInfo(const PointENU &location, const double radius,
                               const std::vector<PointENU> &points,
                               std::vector<PointInfo> *const points_info) {
  // For localization offset
  PointENU real_location = location;
  real_location.set_x(location.x() + location.offset_x());
  real_location.set_y(location.y() + location.offset_y());
  real_location.set_offset_x(0);
  real_location.set_offset_y(0);
  std::vector<PointENU> real_points(points);
  for (auto &point : real_points) {
    point.set_x(point.x() + location.offset_x());
    point.set_y(point.y() + location.offset_y());
  }
  auto status = map_client_.GetPointOnRoad(
      real_points, real_location, points_info, radius, kMapClientTimeoutMs);
  if (!status.ok()) {
    AERROR_EVERY(100) << "Failed to call map service GetPointOnRoad.";
    return false;
  }
  if (points.size() != points_info->size()) {
    AERROR << "The return value of hdmap is wrong";
    return false;
  }

  return true;
}

bool HDMapInput::GetLocalMapElements(
    const PointENU &location, const double radius,
    roadstar::hdmap::MapElements *map_elements) {
  roadstar::hdmap::MapProto map;
  auto status = map_client_.GetLocalMap(location.x(), location.y(), radius,
                                        &map, kMapClientTimeoutMs);
  if (!status.ok()) {
    AERROR_EVERY(100) << "Failed to call map service GetLocalMap";
    return false;
  }
  std::vector<roadstar::hdmap::MapUnit> map_units;
  for (const auto &unit : map.map_units()) {
    map_units.push_back(unit);
  }
  status = map_client_.RetrieveMapElements(map_units, map_elements,
                                           kMapClientTimeoutMs);
  if (!status.ok()) {
    AERROR_EVERY(100) << "Failed to call map service RetrieveMapElements";
    return false;
  }
  return true;
}

bool HDMapInput::GetLocalLanemarkers(const PointENU &location,
                                     const double forward_distance,
                                     const double backward_distance,
                                     std::vector<Lanemarkers> *lanemarkers,
                                     const int map_client_timeout_ms) {
  auto status = map_client_.GetLocalLanemarkers(
      location.x(), location.y(), forward_distance, backward_distance,
      lanemarkers, kMapClientTimeoutMs);

  if (!status.ok()) {
    AERROR_EVERY(100) << "Failed to call map service RetrieveMapElements";
    return false;
  }
  return true;
}

bool HDMapInput::GetCurrMap(std::string *map_name) {
  auto status = map_client_.GetCurrMap(map_name, 100);

  if (!status.ok()) {
    AERROR_EVERY(100) << "Failed to call map service GetCurrMap";
    return false;
  }

  return true;
}

bool HDMapInput::GetLocalTrafficLight(
    const PointENU &location, const double forward_distance,
    roadstar::hdmap::TrafficLight *const traffic_light) {
  roadstar::hdmap::Path path;
  auto status =
      map_client_.GetLocalPath(location.x(), location.y(), forward_distance, 0,
                               &path, kMapClientTimeoutMs);
  if (!status.ok()) {
    AERROR_EVERY(100) << "Failed to get local traffic light";
    return false;
  }
  std::vector<roadstar::hdmap::MapUnit> map_units;
  for (const auto &unit : path.path_units()) {
    map_units.push_back(unit);
  }
  roadstar::hdmap::MapElements map_elements;
  status = map_client_.RetrieveMapElements(map_units, &map_elements,
                                           kMapClientTimeoutMs);
  if (!status.ok()) {
    AERROR_EVERY(100) << "Failed to call map service RetrieveMapElements";
    return false;
  }

  for (int i = 0; i < path.path_units_size() - 2; ++i) {
    const auto &unit = path.path_units(i);
    if (unit.type() == MapUnit::MAP_UNIT_SECTION) {
      const auto &section = map_elements.sections().at(unit.id());
      // Find a section has a traffic light.
      if (section.traffic_light_size() > 0) {
        const auto &next_section_unit = path.path_units(i + 2);
        const auto &next_section =
            map_elements.sections().at(next_section_unit.id());
        double min_dist = std::numeric_limits<double>::max();
        int min_index = 0;
        roadstar::common::math::Polygon2d next_section_polygon(
            next_section.polygon());
        for (int j = 0; j < section.traffic_light_size(); ++j) {
          const auto &traffic_light = section.traffic_light(j);
          roadstar::common::math::Vec2d tl_position(
              traffic_light.position().x(), traffic_light.position().y());
          double dist = next_section_polygon.DistanceTo(tl_position);
          if (dist < min_dist) {
            min_dist = dist;
            min_index = j;
          }
        }
        *traffic_light = section.traffic_light(min_index);
        return true;
      }
    }
  }

  return false;
}

bool HDMapInput::GetCurrMapRoute(std::string *map_route_name) {
  std::string map_name, route_name;
  auto status_map = map_client_.GetCurrMap(&map_name, 100);
  auto status_route = map_client_.GetCurrRoute(&route_name, 100);

  if (!status_map.ok() || !status_route.ok()) {
    AERROR_EVERY(100) << "Failed to call map service GetCurrMap";
    return false;
  }
  *map_route_name = map_name + "/" + route_name;
  return true;
}

void HDMapInput::MergeLanemarker(
    hdmap::MergedLanemarker *prev_merged_lanemarker,
    hdmap::MergedLanemarker *next_merged_lanemarker,
    std::unordered_map<int, hdmap::MergedLanemarker *> *merged_lanemarkers) {
  prev_merged_lanemarker->mutable_curve()->mutable_points()->MergeFrom(
      next_merged_lanemarker->curve().points());
  if (prev_merged_lanemarker->has_type() &&
      prev_merged_lanemarker->type() != next_merged_lanemarker->type()) {
    prev_merged_lanemarker->clear_type();
  }
  for (const auto &id : next_merged_lanemarker->ids()) {
    merged_lanemarkers->insert({id, prev_merged_lanemarker});
  }
}

void HDMapInput::MergeLanemarker(
    hdmap::MergedLanemarker *merged_lanemarker, hdmap::Lanemarker *lanemarker,
    std::unordered_map<int, hdmap::MergedLanemarker *> *merged_lanemarkers,
    bool back) {
  merged_lanemarker->add_ids(lanemarker->id());
  if (merged_lanemarker->has_type() &&
      merged_lanemarker->type() != lanemarker->type()) {
    merged_lanemarker->clear_type();
  }
  if (back) {
    if (merged_lanemarker->curve().points_size() == 0) {
      merged_lanemarker->mutable_curve()->mutable_points()->Swap(
          lanemarker->mutable_curve()->mutable_points());
    } else {
      merged_lanemarker->mutable_curve()->mutable_points()->MergeFrom(
          lanemarker->curve().points());
    }
  } else {
    auto old_points =
        std::move(*merged_lanemarker->mutable_curve()->mutable_points());
    merged_lanemarker->mutable_curve()->mutable_points()->Swap(
        lanemarker->mutable_curve()->mutable_points());
    merged_lanemarker->mutable_curve()->mutable_points()->MergeFrom(old_points);
  }
  merged_lanemarkers->insert({lanemarker->id(), merged_lanemarker});
}

bool HDMapInput::GetLocalMergedLanemarkers(
    const roadstar::common::PointENU &location, const double forward_distance,
    const double backward_distance,
    std::vector<hdmap::MergedLanemarker> *lanemarkers,
    const int map_client_timeout_ms) {
  hdmap::Path path;

  auto status =
      map_client_.GetLocalPath(location.x(), location.y(), forward_distance,
                               backward_distance, &path, map_client_timeout_ms);

  if (!status.ok()) {
    AERROR_EVERY(100) << "Failed to call map service GetLocalPath";
    return false;
  }

  hdmap::MapElements map_elements;
  status = map_client_.RetrieveMapElements(
      {path.path_units().begin(), path.path_units().end()}, &map_elements,
      map_client_timeout_ms);

  if (!status.ok()) {
    AERROR_EVERY(100) << "Failed to call map service RetrieveMapElements";
    return false;
  }

  auto &connections = *map_elements.mutable_connections();
  auto &sections = *map_elements.mutable_sections();

  lanemarkers->clear();
  lanemarkers->reserve(path.path_units_size());

  std::unordered_map<int, hdmap::MergedLanemarker *> merged_lanemarkers;

  for (const auto &path_uint : path.path_units()) {
    if (path_uint.type() == MapUnit::MAP_UNIT_CONNECTION) {
      auto &connection = connections[path_uint.id()];
      int prev_section_id = -1, next_section_id = -1;
      for (const auto &turn_info : connection.turn_info()) {
        if (!turn_info.has_type() ||
            turn_info.type() != hdmap::TurnType::NO_TURN) {
          continue;
        }
        if (turn_info.has_prev_section_id()) {
          if (sections.count(turn_info.prev_section_id()) > 0) {
            if (prev_section_id == -1) {
              prev_section_id = turn_info.prev_section_id();
            } else {
              prev_section_id = -1;  // More than one prev section
              break;
            }
          }
        }
        if (turn_info.has_next_section_id()) {
          if (sections.count(turn_info.next_section_id()) > 0) {
            if (next_section_id == -1) {
              next_section_id = turn_info.next_section_id();
            } else {
              next_section_id = -1;  // More than one next section
              break;
            }
          }
        }
      }

      if (prev_section_id == -1 || next_section_id == -1) {
        continue;
      }

      auto &prev_section = sections[prev_section_id],
           &next_section = sections[next_section_id];
      if (prev_section.lanemarkers_size() != next_section.lanemarkers_size()) {
        continue;
      }

      for (int i = 0; i < prev_section.lanemarkers_size(); ++i) {
        auto &prev_lanemarker = *prev_section.mutable_lanemarkers(i),
             &next_lanemarker = *next_section.mutable_lanemarkers(i);
        auto prev_merged_iter = merged_lanemarkers.find(prev_lanemarker.id());
        auto next_merged_iter = merged_lanemarkers.find(next_lanemarker.id());
        auto end = merged_lanemarkers.end();

        if (prev_merged_iter == end && next_merged_iter == end) {
          auto *merged_lanemarker = new hdmap::MergedLanemarker();
          merged_lanemarker->set_type(prev_lanemarker.type());
          MergeLanemarker(merged_lanemarker, &prev_lanemarker,
                          &merged_lanemarkers, true);
          MergeLanemarker(merged_lanemarker, &next_lanemarker,
                          &merged_lanemarkers, true);
        } else if (next_merged_iter == end) {
          MergeLanemarker(merged_lanemarkers[prev_lanemarker.id()],
                          &next_lanemarker, &merged_lanemarkers, true);
        } else if (prev_merged_iter == end) {
          MergeLanemarker(merged_lanemarkers[next_lanemarker.id()],
                          &prev_lanemarker, &merged_lanemarkers, false);
        } else {
          auto *next = merged_lanemarkers[next_lanemarker.id()];
          MergeLanemarker(merged_lanemarkers[prev_lanemarker.id()], next,
                          &merged_lanemarkers);
          delete next;
        }
      }
    }
  }

  int idx = 0;
  for (const auto &path_unit : path.path_units()) {
    if (path_unit.type() == MapUnit::MAP_UNIT_SECTION) {
      for (auto &lm : *sections[path_unit.id()].mutable_lanemarkers()) {
        hdmap::MergedLanemarker l;
        if (merged_lanemarkers.find(lm.id()) != merged_lanemarkers.end()) {
          continue;
        }
        l.add_ids(lm.id());
        l.set_id(idx++);
        l.set_type(lm.type());
        l.mutable_curve()->Swap(lm.mutable_curve());
        lanemarkers->emplace_back(std::move(l));
      }
    }
  }
  std::unordered_set<hdmap::MergedLanemarker *> pushed_lanemarkers;

  for (auto &pair : merged_lanemarkers) {
    auto &merged_lanemarker = pair.second;
    if (pushed_lanemarkers.find(merged_lanemarker) !=
        pushed_lanemarkers.end()) {
      continue;
    }
    merged_lanemarker->set_id(idx++);
    pushed_lanemarkers.insert(merged_lanemarker);
    lanemarkers->emplace_back(std::move(*merged_lanemarker));
  }

  for (auto &merged_lanemarker : pushed_lanemarkers) {
    delete merged_lanemarker;
  }
  return true;
}

}  // namespace common
}  // namespace roadstar
