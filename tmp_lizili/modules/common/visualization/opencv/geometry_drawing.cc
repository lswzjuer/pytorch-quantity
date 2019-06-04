#include "modules/common/visualization/opencv/geometry_drawing.h"

#include <vector>

namespace roadstar {
namespace common {
namespace visualization {

void DrawPolygon(const ::roadstar::common::Polygon &polygon,
                 const CanvasTransform &trans, cv::Mat *mat,
                 ::roadstar::common::util::Color color, double width) {
  std::vector<cv::Point> contour;
  contour.reserve(polygon.points().size());
  for (const auto &point : polygon.points()) {
    contour.emplace_back(trans.GetImgPoint(point.x(), point.y()));
  }
  const cv::Point *pts =
      reinterpret_cast<const cv::Point *>(cv::Mat(contour).data);
  int npts = cv::Mat(contour).rows;
  cv::polylines(*mat, &pts, &npts, 1, true, color.cv_rgb(), width);
}

void DrawCurve(const ::roadstar::common::Curve &curve,
               const CanvasTransform &trans, cv::Mat *mat,
               ::roadstar::common::util::Color color, double width) {
  std::vector<cv::Point> contour;
  contour.reserve(curve.points().size());
  for (const auto &point : curve.points()) {
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
