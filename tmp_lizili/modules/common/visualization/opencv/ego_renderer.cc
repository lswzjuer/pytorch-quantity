#include "modules/common/visualization/opencv/ego_renderer.h"

#include <vector>

#include "modules/common/coordinate/coord_trans.h"
#include "modules/common/visualization/opencv/geometry_drawing.h"

namespace roadstar {
namespace common {
namespace visualization {

void EgoRenderer::DoRender(
    const ::roadstar::localization::Localization& localization,
    const CanvasTransform& trans, cv::Mat* mat) {
  DrawEgoBox(localization, trans, mat, param_.ego_box_width(),
             param_.ego_box_length(), param_.ego_periphery_color(),
             param_.ego_periphery_width());
}

void DrawEgoBox(const ::roadstar::localization::Localization& localization,
                const CanvasTransform& trans, cv::Mat* mat,
                double ego_box_width, double ego_box_length,
                ::roadstar::common::util::Color color, double line_width) {
  std::vector<double> xs{ego_box_length / 2, ego_box_length / 2,
                         -ego_box_length / 2, -ego_box_length / 2};
  std::vector<double> ys{-ego_box_width / 2, ego_box_width / 2,
                         ego_box_width / 2, -ego_box_width / 2};
  Eigen::Matrix<double, 3, 1> translation;
  translation(0) = localization.utm_x();
  translation(1) = localization.utm_y();
  translation(2) = 0;
  CoordTransD to_utm(localization.heading(), translation);
  std::vector<Eigen::Vector3d> contour{
      to_utm.TransformCoord3d(Eigen::Vector3d(xs[0], ys[0], 0)),
      to_utm.TransformCoord3d(Eigen::Vector3d(xs[1], ys[1], 0)),
      to_utm.TransformCoord3d(Eigen::Vector3d(xs[2], ys[2], 0)),
      to_utm.TransformCoord3d(Eigen::Vector3d(xs[3], ys[3], 0))};
  DrawPolygon(contour, trans, mat, color, line_width);
}

}  // namespace visualization
}  // namespace common
}  // namespace roadstar
