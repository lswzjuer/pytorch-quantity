#include "modules/common/hdmap/base/road_edge_object.h"

namespace roadstar {
namespace common {
namespace hdmap {

RoadEdgeObject::RoadEdgeObject(::roadstar::hdmap::RoadEdge &&road_edge)
    : road_edge_(std::move(road_edge)) {
  polygon_ = ::roadstar::common::math::Polygon2d(road_edge_.polygon());
}

}  // namespace hdmap
}  // namespace common
}  // namespace roadstar
