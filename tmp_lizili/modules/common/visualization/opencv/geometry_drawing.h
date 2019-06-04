#ifndef MODULES_COMMON_VISUALIZATION_OPENCV_GEOMETRY_DRAWING_H_
#define MODULES_COMMON_VISUALIZATION_OPENCV_GEOMETRY_DRAWING_H_

#include <vector>

#include "opencv2/opencv.hpp"

#include "modules/common/proto/geometry.pb.h"
#include "modules/common/util/colormap.h"
#include "modules/common/visualization/opencv/canvas_transform.h"

namespace roadstar {
namespace common {
namespace visualization {

void DrawPolygon(const ::roadstar::common::Polygon &polygon,
                 const CanvasTransform &trans, cv::Mat *mat,
                 ::roadstar::common::util::Color color =
                     ::roadstar::common::util::ColorName::Blue,
                 double width = 1.0);

void DrawCurve(const ::roadstar::common::Curve &curve,
               const CanvasTransform &trans, cv::Mat *mat,
               ::roadstar::common::util::Color color =
                   ::roadstar::common::util::ColorName::Green,
               double width = 1.0);

/**
 * NOTE:
 *   1. points must be iterable
 *   2. points must have field size()
 *   3. point must have fields x() and y()
 */
template <typename PointContainer>
void DrawPolygon(const PointContainer &points, const CanvasTransform &trans,
                 cv::Mat *mat,
                 ::roadstar::common::util::Color color =
                     ::roadstar::common::util::ColorName::Blue,
                 double width = 1.0) {
  std::vector<cv::Point> contour;
  contour.reserve(points.size());
  for (const auto &point : points) {
    contour.emplace_back(trans.GetImgPoint(point.x(), point.y()));
  }
  const cv::Point *pts =
      reinterpret_cast<const cv::Point *>(cv::Mat(contour).data);
  int npts = cv::Mat(contour).rows;
  cv::polylines(*mat, &pts, &npts, 1, true, color.cv_rgb(), width);
}

/**
 * NOTE:
 *   1. points must be iterable
 *   2. points must have field size()
 *   3. point must have fields x() and y()
 */
template <typename PointContainer>
void DrawCurve(const PointContainer &points, const CanvasTransform &trans,
               cv::Mat *mat,
               ::roadstar::common::util::Color color =
                   ::roadstar::common::util::ColorName::Green,
               double width = 1.0) {
  std::vector<cv::Point> contour;
  contour.reserve(points.size());
  for (const auto &point : points) {
    contour.emplace_back(trans.GetImgPoint(point.x(), point.y()));
  }
  const cv::Point *pts =
      reinterpret_cast<const cv::Point *>(cv::Mat(contour).data);
  int npts = cv::Mat(contour).rows;
  cv::polylines(*mat, &pts, &npts, 1, false, color.cv_rgb(), width);
}

}  // namespace visualization
}  // namespace common
}  // namespace roadstar

#endif
