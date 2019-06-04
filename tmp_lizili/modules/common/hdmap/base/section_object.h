#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "modules/common/hdmap/base/map_object.h"
#include "modules/common/math/aaboxkdtree2d.h"
#include "modules/common/math/polygon2d.h"
#include "modules/msgs/hdmap/proto/section.pb.h"

namespace roadstar {
namespace common {
namespace hdmap {

class SectionObject;
template <>
struct Accumulator<SectionObject> {
  static constexpr ::roadstar::hdmap::MapUnit::MapUnitType kType =
      ::roadstar::hdmap::MapUnit::MAP_UNIT_SECTION;
};

class SectionObject : public MapObject {
 public:
  explicit SectionObject(::roadstar::hdmap::Section &&section);

  const ::roadstar::hdmap::Section &section() const {
    return section_;
  }

  const ::roadstar::common::math::Polygon2d &polygon() const override {
    return polygon_;
  }

  ::roadstar::hdmap::MapUnit map_unit() const override {
    ::roadstar::hdmap::MapUnit map_unit;
    map_unit.set_id(section_.id());
    map_unit.set_type(Accumulator<SectionObject>::kType);

    return map_unit;
  }

  double length() const {
    return section_.length();
  }

 private:
  // Section data.
  ::roadstar::hdmap::Section section_;

  // Section polygon.
  ::roadstar::common::math::Polygon2d polygon_;

 private:
  void MakeLanePolygon();

  void MakeEndPoints();
};

}  // namespace hdmap
}  // namespace common
}  // namespace roadstar
