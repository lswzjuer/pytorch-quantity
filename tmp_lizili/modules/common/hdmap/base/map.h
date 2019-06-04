#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "modules/common/hdmap/base/connection_object.h"
#include "modules/common/hdmap/base/id.h"
#include "modules/common/hdmap/base/map_unit_hash.h"
#include "modules/common/hdmap/base/road_edge_object.h"
#include "modules/common/hdmap/base/section_object.h"

namespace roadstar {
namespace common {
namespace hdmap {

/**
 * @brief BaseMap provide a base map implementation.
 *
 * */
class BaseMap {
 public:
  BaseMap() = default;

  virtual ~BaseMap() = default;

  BaseMap(const BaseMap &other) : map_objects_(other.map_objects_) {}

  BaseMap &operator=(const BaseMap &other) {
    map_objects_ = other.map_objects_;
    return *this;
  }

  /**
   * @brief Clear the map.
   * */
  void Clear() {
    map_objects_.clear();
  }

  void Insert(const ConstMapObjectPtr &map_object) {
    map_objects_.emplace(map_object->map_unit(), map_object);
  }

  template <typename ObjectType>
  void Insert(const typename std::enable_if<
              std::is_base_of<MapObject, ObjectType>::value, ObjectType>::type
                  &object) {
    auto map_unit = object.map_unit();
    map_objects_.emplace(map_unit, std::make_shared<ObjectType>(object));
  }

  template <typename ObjectType>
  void Insert(
      typename std::enable_if<std::is_base_of<MapObject, ObjectType>::value,
                              ObjectType>::type &&object) {
    auto map_unit = object.map_unit();
    map_objects_.emplace(map_unit,
                         std::make_shared<ObjectType>(std::move(object)));
  }

  /**
   * @brief Check the map has the map unit.
   * @return A bool value.
   * */
  bool HasMapUnit(const ::roadstar::hdmap::MapUnit &map_unit) const {
    return map_objects_.find(map_unit) != map_objects_.end();
  }

  ConstMapObjectPtr GetMapObject(
      const ::roadstar::hdmap::MapUnit &map_unit) const {
    auto it = map_objects_.find(map_unit);
    return it != map_objects_.end() ? it->second : nullptr;
  }

  /**
   * @brief Remove a specific map unit.
   * @return A bool value indicates whether the removal took place. True for
   * Removal, False for No Removal.
   * */
  bool RemoveMapUnit(const ::roadstar::hdmap::MapUnit &map_unit) {
    return map_objects_.erase(map_unit);
  }

  const std::unordered_map<::roadstar::hdmap::MapUnit, ConstMapObjectPtr>
      &map_objects() const {
    return map_objects_;
  }

 protected:
  std::unordered_map<::roadstar::hdmap::MapUnit, ConstMapObjectPtr> map_objects_;
};  // class BaseMap

}  // namespace hdmap
}  // namespace common
}  // namespace roadstar
