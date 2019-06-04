#include "modules/common/visualization/opencv/fusion_map_renderer.h"

#include <cstring>
#include <string>
#include <vector>

#include "modules/common/coordinate/coord_trans.h"
#include "modules/common/visualization/opencv/geometry_drawing.h"

namespace roadstar {
namespace common {
namespace visualization {

void DrawObstacleId(const ::roadstar::perception::Obstacle& obstacle,
                    const CanvasTransform& trans, cv::Mat* mat,
                    ::roadstar::common::util::Color color =
                        ::roadstar::common::util::ColorName::Green,
                    double size = 1.0);

void FusionMapRenderer::DoRender(
    const ::roadstar::perception::FusionMap& fusion_map,
    const CanvasTransform& trans, cv::Mat* mat) {
  for (const auto& obstacle : fusion_map.obstacles()) {
    DrawObstacleBox(obstacle, trans, mat, param_.obstacle_periphery_color(),
                    param_.obstacle_periphery_width());
  }
  if (param_.with_obstacle_id()) {
    for (const auto& obstacle : fusion_map.obstacles()) {
      DrawObstacleId(obstacle, trans, mat, param_.obstacle_id_text_color(),
                     param_.obstacle_id_text_size());
    }
  }
}

void DrawObstacleBox(const ::roadstar::perception::Obstacle& obstacle,
                     const CanvasTransform& trans, cv::Mat* mat,
                     ::roadstar::common::util::Color color, double width) {
  std::vector<double> xs{obstacle.length() / 2, obstacle.length() / 2,
                         -obstacle.length() / 2, -obstacle.length() / 2};
  std::vector<double> ys{-obstacle.width() / 2, obstacle.width() / 2,
                         obstacle.width() / 2, -obstacle.width() / 2};
  Eigen::Matrix<double, 3, 1> translation;
  translation(0) = obstacle.position().x();
  translation(1) = obstacle.position().y();
  translation(2) = 0;
  CoordTransD to_utm(obstacle.theta(), translation);
  std::vector<Eigen::Vector3d> contour{
      to_utm.TransformCoord3d(Eigen::Vector3d(xs[0], ys[0], 0)),
      to_utm.TransformCoord3d(Eigen::Vector3d(xs[1], ys[1], 0)),
      to_utm.TransformCoord3d(Eigen::Vector3d(xs[2], ys[2], 0)),
      to_utm.TransformCoord3d(Eigen::Vector3d(xs[3], ys[3], 0))};
  DrawPolygon(contour, trans, mat, color, width);
}

void DrawObstacleId(const ::roadstar::perception::Obstacle& obstacle,
                    const CanvasTransform& trans, cv::Mat* mat,
                    ::roadstar::common::util::Color color, double size) {
  auto pt = trans.GetImgPoint(obstacle.position().x(), obstacle.position().y());
  cv::putText(*mat, std::to_string(obstacle.track_id()), pt,
              cv::FONT_HERSHEY_SIMPLEX, size, color.cv_rgb());
}

}  // namespace visualization
}  // namespace common
}  // namespace roadstar
