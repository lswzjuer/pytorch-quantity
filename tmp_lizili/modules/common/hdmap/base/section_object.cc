#include "modules/common/hdmap/base/section_object.h"

namespace roadstar {
namespace common {
namespace hdmap {

SectionObject::SectionObject(::roadstar::hdmap::Section &&section)
    : section_(std::move(section)) {
  polygon_ = ::roadstar::common::math::Polygon2d(section_.polygon());
}

}  // namespace hdmap
}  // namespace common
}  // namespace roadstar
