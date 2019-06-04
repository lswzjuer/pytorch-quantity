#include "modules/common/visualization/util.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace roadstar {
namespace common {
namespace visualization {

using roadstar::common::PointENU;
using roadstar::common::adapter::AdapterManager;
using roadstar::common::util::Color;
using roadstar::common::util::ColorName;

cv::Point CanvasTrans::TransUTM(double utm_x, double utm_y) const {
  Eigen::Vector3d coord;
  coord << utm_x, utm_y, 0;
  Eigen::Vector3d ego_coord = world_imu_trans.TransformCoord3d(coord);
  return TransEgo(ego_coord.x(), ego_coord.y());
}

cv::Point CanvasTrans::TransEgo(double ego_x, double ego_y) const {
  int img_x = -(ego_y + offset_x) * scale_x + w / 2;
  int img_y = -(ego_x + offset_y) * scale_y + h / 2;
  return cv::Point(img_x, img_y);
}

MapRenderer::MapRenderer(MapRendererParam param)
    : hdmap_radius_(param.hdmap_radius()),
      lane_marker_width_(param.lane_marker_width()),
      lane_marker_color_(Color(param.lane_marker_color()).cv_rgb()),
      vehicle_length_(param.vehicle_length()),
      vehicle_width_(param.vehicle_width()),
      vehicle_color_(Color(param.vehicle_color()).cv_rgb()),
      draw_grid_lines_(param.draw_grid_lines()),
      grid_renderer_(param.grid_renderer_param()) {
  hdmap_input_ = HDMapInput::instance();
}

void MapRenderer::RenderMap(const Localization &loc, const CanvasTrans &trans,
                            cv::Mat *map_img, const cv::Scalar &bg_color) {
  // draw map
  *map_img = cv::Mat(trans.h, trans.w, CV_8UC3, bg_color);

  // draw grid lines
  if (draw_grid_lines_) {
    grid_renderer_.DrawGridLines(trans, map_img);
  }
  // draw map
  DrawMap(loc, trans, map_img);
  // draw self
  cv::circle(*map_img, trans.TransUTM(loc.utm_x(), loc.utm_y()), 5,
             vehicle_color_, -1);
  std::vector<cv::Point> rect;
  GetObjectRenderRect(loc.utm_x(), loc.utm_y(), vehicle_length_, vehicle_width_,
                      loc.heading(), trans, &rect);
  DrawRect(rect, vehicle_color_, map_img);
}

void MapRenderer::DrawLaneMarker(const roadstar::hdmap::Lanemarker &lane_marker,
                                 double offset_x, double offset_y,
                                 const CanvasTrans &trans, cv::Mat *mat) {
  std::vector<cv::Point> contour;
  for (const auto &point : lane_marker.curve().points()) {
    contour.emplace_back(
        trans.TransUTM(point.x() - offset_x, point.y() - offset_y));
  }
  const cv::Point *pts =
      reinterpret_cast<const cv::Point *>(cv::Mat(contour).data);
  int npts = cv::Mat(contour).rows;
  cv::polylines(*mat, &pts, &npts, 1, false, lane_marker_color_,
                lane_marker_width_);
}

void MapRenderer::DrawConnection(const roadstar::hdmap::Connection &connection,
                                 double offset_x, double offset_y,
                                 const CanvasTrans &trans, cv::Mat *mat) {
  std::vector<cv::Point> contour;
  for (const auto &point : connection.polygon().points()) {
    contour.emplace_back(
        trans.TransUTM(point.x() - offset_x, point.y() - offset_y));
  }
  const cv::Point *pts =
      reinterpret_cast<const cv::Point *>(cv::Mat(contour).data);
  int npts = cv::Mat(contour).rows;
  cv::polylines(*mat, &pts, &npts, 1, false, lane_marker_color_,
                lane_marker_width_);
}

void MapRenderer::DrawMap(const Localization &loc, const CanvasTrans &trans,
                          cv::Mat *mat) {
  roadstar::hdmap::MapElements map_elements;
  PointENU location;
  location.set_x(loc.utm_x() + loc.offset_utm_x());
  location.set_y(loc.utm_y() + loc.offset_utm_y());
  hdmap_input_->GetLocalMapElements(location, hdmap_radius_, &map_elements);
  auto sec_iter = map_elements.sections().begin();
  for (sec_iter = map_elements.sections().begin();
       sec_iter != map_elements.sections().end(); sec_iter++) {
    for (const auto &lane_marker : sec_iter->second.lanemarkers()) {
      DrawLaneMarker(lane_marker, loc.offset_utm_x(), loc.offset_utm_y(), trans,
                     mat);
    }
  }
  auto connection_iter = map_elements.connections().begin();
  for (connection_iter = map_elements.connections().begin();
       connection_iter != map_elements.connections().end(); connection_iter++) {
    DrawConnection(connection_iter->second, loc.offset_utm_x(),
                   loc.offset_utm_y(), trans, mat);
  }
}

GridRenderer::GridRenderer(GridRendererParam param)
    : grid_cell_height_(param.grid_cell_height()),
      grid_cell_width_(param.grid_cell_width()),
      grid_bold_gap_(param.grid_bold_gap()),
      grid_line_width_(param.grid_line_width()),
      grid_line_color_(Color(param.grid_line_color()).cv_rgb()) {}

void GridRenderer::DrawGridLines(const CanvasTrans &trans, cv::Mat *mat) {
  const auto &color = grid_line_color_;
  double half_width = trans.view_w / 2;
  double half_height = trans.view_h / 2;

  // draw horizontal lines
  for (int i = 0; i * grid_cell_height_ <= half_height; ++i) {
    double x = i * grid_cell_height_;
    cv::Point start_p = trans.TransEgo(x, -half_width);
    cv::Point end_p = trans.TransEgo(x, half_width);
    double thickness = grid_line_width_;
    bool bold =
        std::fmod(x, grid_bold_gap_) == 0 && grid_cell_height_ < grid_bold_gap_;
    if (bold) {
      thickness *= 2;
      cv::putText(*mat, std::to_string(static_cast<int>(x)), end_p, 1, 1, color,
                  thickness);
    }
    cv::line(*mat, start_p, end_p, color, thickness);
    start_p = trans.TransEgo(-x, -half_width);
    end_p = trans.TransEgo(-x, half_width);
    cv::line(*mat, start_p, end_p, color, thickness);
    if (bold) {
      cv::putText(*mat, std::to_string(static_cast<int>(-x)), end_p, 1, 1,
                  color, thickness);
    }
  }

  // draw vertical lines
  for (int i = 0; i * grid_cell_width_ <= half_width; i++) {
    double y = i * grid_cell_width_;
    cv::Point start_p = trans.TransEgo(half_height, y);
    cv::Point end_p = trans.TransEgo(-half_height, y);
    double thickness = grid_line_width_;
    bool bold =
        std::fmod(y, grid_bold_gap_) == 0 && grid_cell_width_ < grid_bold_gap_;
    if (bold) {
      thickness *= 2;
      cv::putText(*mat, std::to_string(static_cast<int>(y)), end_p, 1, 1, color,
                  thickness);
    }
    cv::line(*mat, start_p, end_p, color, thickness);
    start_p = trans.TransEgo(half_height, -y);
    end_p = trans.TransEgo(-half_height, -y);
    cv::line(*mat, start_p, end_p, color, thickness);
    if (bold) {
      cv::putText(*mat, std::to_string(static_cast<int>(-y)), end_p, 1, 1,
                  color, thickness);
    }
  }
}  // namespace visualization

CvRenderer::CvRenderer() {}

void GetObjectRenderRect(double center_x, double center_y, double length,
                         double width, double theta, const CanvasTrans &trans,
                         std::vector<cv::Point> *rect) {
  if (width > 0) {
    std::vector<Eigen::Vector2d> points = GetCornerPoints(
        Eigen::Vector2d(center_x, center_y), length, width, theta);
    for (const auto &point : points) {
      rect->emplace_back(trans.TransUTM(point.x(), point.y()));
    }
  }
}

void DrawRect(const std::vector<cv::Point> &rect, const cv::Scalar &color,
              cv::Mat *mat) {
  for (size_t i = 0; i < rect.size(); i++) {
    cv::line(*mat, rect[i], rect[(i + 1) % rect.size()], color);
  }
}

std::vector<Eigen::Vector2d> GetCornerPoints(const Eigen::Vector2d &center,
                                             const double length,
                                             const double width,
                                             const double theta) {
  std::vector<Eigen::Vector2d> points;
  double l_half = length / 2;
  double w_half = width / 2;
  Eigen::Vector2d v(l_half * cos(theta), l_half * sin(theta));
  Eigen::Vector2d u(-w_half * sin(theta), w_half * cos(theta));
  // corner points
  points.emplace_back(center + v + u);
  points.emplace_back(center - v + u);
  points.emplace_back(center - v - u);
  points.emplace_back(center + v - u);
  return points;
}

}  // namespace visualization
}  // namespace common
}  // namespace roadstar
