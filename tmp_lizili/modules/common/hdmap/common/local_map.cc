#include "modules/common/hdmap/common/local_map.h"

#include <memory>
#include <utility>
#include <vector>

#include "modules/common/hdmap/client/map_client.h"

namespace roadstar {
namespace common {
namespace hdmap {

namespace {

// The client timeout.
constexpr size_t kMapClientTimeoutMs = 1000;

}  // namespace

LocalMap::LocalMap() : radius_(0) {
  center_.set_x(0);
  center_.set_y(0);
  center_.set_z(0);
}

bool LocalMap::Update(const ::roadstar::common::PointENU &center, double radius) {
  MapClient client;
  ::roadstar::hdmap::MapProto map;
  auto get_local_map_status =
      client.GetLocalMap(center, radius, &map, kMapClientTimeoutMs);
  if (!get_local_map_status.ok()) {
    AERROR << "Failed to call GetLocalMap on " << center.x() << " "
           << center.y() << " with radius " << radius << " since "
           << get_local_map_status.error_message();
    return false;
  }

  // Get the new map units and retrieve their elements and insert them into
  // local map.
  std::vector<::roadstar::hdmap::MapUnit> new_map_units;
  for (const auto &map_unit : map.map_units()) {
    if (!HasMapUnit(map_unit)) {
      new_map_units.push_back(map_unit);
    }
  }
  ::roadstar::hdmap::MapElements elements;
  auto get_map_elements_status =
      client.GetMapElements(new_map_units, &elements, kMapClientTimeoutMs);
  if (!get_map_elements_status.ok()) {
    AERROR << "Failed to call GetMapElements since "
           << get_map_elements_status.error_message();
    return false;
  }
  // Add the new map object.
  for (auto &section : *elements.mutable_sections()) {
    Insert<SectionObject>(SectionObject(std::move(section.second)));
  }
  for (auto &connection : *elements.mutable_connections()) {
    Insert<ConnectionObject>(ConnectionObject(std::move(connection.second)));
  }
  for (auto &road_edge : *elements.mutable_road_edges()) {
    Insert<RoadEdgeObject>(RoadEdgeObject(std::move(road_edge.second)));
  }

  // Get the expired map unit and remove them.
  for (const auto &kval : map_objects_) {
    const auto &curr_map_unit = kval.first;
    bool is_find = false;
    // This may be more efficient using a set.
    // Considering the update is not frequent and the size of map.map_units is
    // small, a linear searching is acceptable.
    for (const auto &map_unit : map.map_units()) {
      if (curr_map_unit.id() == map_unit.id() &&
          curr_map_unit.type() == map_unit.type()) {
        is_find = true;
        break;
      }
    }
    // Not find.
    // The map unit is expired and remove it.
    if (!is_find) {
      RemoveMapUnit(curr_map_unit);
    }
  }

  center_ = center;
  radius_ = radius;

  return true;
}

void LocalMap::Shrink(const ::roadstar::common::PointENU &center, double radius) {
  ::roadstar::common::math::Vec2d vec2d_center(center.x(), center.y());
  for (auto it = map_objects_.begin(); it != map_objects_.end();) {
    const auto &object = it->second;
    if (object->polygon().DistanceTo(vec2d_center) > radius) {
      it = map_objects_.erase(it);
    } else {
      ++it;
    }
  }
}

ConstMapObjectPtr LocalMap::GetMapObjectByPoint(
    const ::roadstar::common::PointENU &point) const {
  for (const auto &kval : map_objects_) {
    if (kval.second->polygon().IsPointIn(
            ::roadstar::common::math::Vec2d(point.x(), point.y()))) {
      return kval.second;
    }
  }

  return nullptr;
}

}  // namespace hdmap
}  // namespace common
}  // namespace roadstar
