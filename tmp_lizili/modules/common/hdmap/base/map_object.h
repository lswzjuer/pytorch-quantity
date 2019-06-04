#pragma once

#include <memory>

#include "modules/common/hdmap/base/id.h"
#include "modules/common/math/aaboxkdtree2d.h"
#include "modules/common/math/polygon2d.h"
#include "modules/msgs/hdmap/proto/hdmap_common.pb.h"

namespace roadstar {
namespace common {
namespace hdmap {

template <typename T>
struct Accumulator;

class MapObject {
 public:
  /**
   * @brief Return the polygon of the object.
   * */
  virtual const ::roadstar::common::math::Polygon2d &polygon() const = 0;

  /**
   * @brief Return the map unit (type and id) of the object.
   * */
  virtual ::roadstar::hdmap::MapUnit map_unit() const = 0;

  /**
   * @brief Return the id of the object.
   * */
  Id id() const {
    return this->map_unit().id();
  }

  /**
   * @brief Return the type of the object.
   * */
  ::roadstar::hdmap::MapUnit::MapUnitType type() const {
    return this->map_unit().type();
  }

  virtual ~MapObject() {}
};
using MapObjectPtr = std::shared_ptr<MapObject>;
using ConstMapObjectPtr = std::shared_ptr<const MapObject>;

}  // namespace hdmap
}  // namespace common
}  // namespace roadstar
