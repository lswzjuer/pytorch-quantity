#pragma once

#include "modules/common/hdmap/base/map.h"

namespace roadstar {
namespace common {
namespace hdmap {

class LocalMap : public BaseMap {
 public:
  LocalMap();

  virtual ~LocalMap() = default;

  LocalMap(const LocalMap &local_map)
      : BaseMap(local_map),
        center_(local_map.center_),
        radius_(local_map.radius_) {}

  bool Update(const ::roadstar::common::PointENU &center, double radius);

  void Shrink(const ::roadstar::common::PointENU &center, double radius);

  const ::roadstar::common::PointENU &center() const {
    return center_;
  }

  double radius() const {
    return radius_;
  }

  ConstMapObjectPtr GetMapObjectByPoint(
      const ::roadstar::common::PointENU &point) const;

 private:
  // The center of local map.
  ::roadstar::common::PointENU center_;

  // The radius of local map.
  double radius_;
};

}  // namespace hdmap
}  // namespace common
}  // namespace roadstar
