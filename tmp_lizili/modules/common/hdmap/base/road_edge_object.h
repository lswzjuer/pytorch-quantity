#pragma once

#include <utility>

#include "modules/common/hdmap/base/map_object.h"
#include "modules/msgs/hdmap/proto/road_edge.pb.h"

namespace roadstar {
namespace common {
namespace hdmap {

class RoadEdgeObject;

template <>
struct Accumulator<RoadEdgeObject> {
  static constexpr ::roadstar::hdmap::MapUnit::MapUnitType kType =
      ::roadstar::hdmap::MapUnit::MAP_UNIT_ROADEDGE;
};

class RoadEdgeObject : public MapObject {
 public:
  explicit RoadEdgeObject(::roadstar::hdmap::RoadEdge &&road_edge);

  const ::roadstar::hdmap::RoadEdge &road_edge() const {
    return road_edge_;
  }

  const ::roadstar::common::math::Polygon2d &polygon() const override {
    return polygon_;
  }

  ::roadstar::hdmap::MapUnit map_unit() const override {
    ::roadstar::hdmap::MapUnit map_unit;
    map_unit.set_id(road_edge_.id());
    map_unit.set_type(Accumulator<RoadEdgeObject>::kType);

    return map_unit;
  }

 private:
  ::roadstar::hdmap::RoadEdge road_edge_;

  ::roadstar::common::math::Polygon2d polygon_;
};

}  // namespace hdmap
}  // namespace common
}  // namespace roadstar
