#include "modules/common/hdmap/base/connection_object.h"

namespace roadstar {
namespace common {
namespace hdmap {

ConnectionObject::ConnectionObject(roadstar::hdmap::Connection &&connection)
    : connection_(std::move(connection)) {
  polygon_ = roadstar::common::math::Polygon2d(connection_.polygon());
}

}  // namespace hdmap
}  // namespace common
}  // namespace roadstar
