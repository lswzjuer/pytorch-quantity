#ifndef MODULES_COMMON_VISUALIZATION_COMMON_CANVAS_TRANSFORM_H_
#define MODULES_COMMON_VISUALIZATION_COMMON_CANVAS_TRANSFORM_H_

#include "Eigen/Geometry"
#include "opencv2/opencv.hpp"

#include "modules/common/coordinate/coord_trans.h"
#include "modules/common/visualization/common/canvas_view.h"

namespace roadstar {
namespace common {
namespace visualization {

inline CoordTransD GetCoordTransFromCanvasToUtm(const CanvasView& view) {
  Eigen::Matrix<double, 3, 1> translation;
  translation(0) = view.utm_x();
  translation(1) = view.utm_y();
  translation(2) = view.utm_z();
  return CoordTransD(view.heading(), translation);
}

inline CoordTransD GetCoordTransFromUtmToCanvas(const CanvasView& view) {
  return GetCoordTransFromCanvasToUtm(view).Inv();
}

class CanvasTransform {
 public:
  explicit CanvasTransform(const CanvasView& view)
      : scale_x_(view.scale_x()),
        scale_y_(view.scale_y()),
        w_(view.w()),
        h_(view.h()),
        coord_trans_(GetCoordTransFromUtmToCanvas(view)) {}
  const CoordTransD& GetCoordTrans() const {
    return coord_trans_;
  }
  cv::Point GetImgPoint(double utm_x, double utm_y) const {
    Eigen::Vector3d coord(utm_x, utm_y, 0);
    Eigen::Vector3d img_point = coord_trans_.TransformCoord3d(coord);
    int img_x = img_point.x() * scale_x_ + w_ / 2.0;
    int img_y = -img_point.y() * scale_y_ + h_ / 2.0;
    return cv::Point(img_x, img_y);
  }

 private:
  const double scale_x_;
  const double scale_y_;
  const int w_;
  const int h_;
  CoordTransD coord_trans_;
};

}  // namespace visualization
}  // namespace common
}  // namespace roadstar

#endif
