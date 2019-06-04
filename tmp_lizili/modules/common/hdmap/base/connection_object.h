#pragma once

#include <memory>
#include <utility>

#include "modules/common/hdmap/base/id.h"
#include "modules/common/hdmap/base/map_object.h"
#include "modules/common/math/polygon2d.h"
#include "modules/msgs/hdmap/proto/connection.pb.h"

namespace roadstar {
namespace common {
namespace hdmap {

class ConnectionObject;
template <>
struct Accumulator<ConnectionObject> {
  static constexpr ::roadstar::hdmap::MapUnit::MapUnitType kType =
      ::roadstar::hdmap::MapUnit::MAP_UNIT_CONNECTION;
};

class ConnectionObject : public MapObject {
 public:
  explicit ConnectionObject(::roadstar::hdmap::Connection &&connection);

  const ::roadstar::hdmap::Connection &connection() const {
    return connection_;
  }

  const ::roadstar::common::math::Polygon2d &polygon() const override {
    return polygon_;
  }

  ::roadstar::hdmap::MapUnit map_unit() const override {
    ::roadstar::hdmap::MapUnit map_unit;
    map_unit.set_id(connection_.id());
    map_unit.set_type(Accumulator<ConnectionObject>::kType);

    return map_unit;
  }

 private:
  ::roadstar::hdmap::Connection connection_;

  ::roadstar::common::math::Polygon2d polygon_;
};

}  // namespace hdmap
}  // namespace common
}  // namespace roadstar
