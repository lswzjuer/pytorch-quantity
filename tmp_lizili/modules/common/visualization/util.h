#ifndef MODULES_COMMON_VISUALIZATION_UTIL_H_
#define MODULES_COMMON_VISUALIZATION_UTIL_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "opencv2/opencv.hpp"

#include "modules/common/adapters/adapter_manager.h"
#include "modules/common/coordinate/sensor_coordinate.h"
#include "modules/common/hdmap_client/hdmap_input.h"
#include "modules/common/log.h"
#include "modules/common/proto/colormap.pb.h"
#include "modules/common/proto/header.pb.h"
#include "modules/common/sensor_source.h"
#include "modules/common/util/colormap.h"
#include "modules/common/visualization/proto/map_renderer.pb.h"
#include "modules/msgs/localization/proto/localization.pb.h"

namespace roadstar {
namespace common {
namespace visualization {

using roadstar::common::adapter::AdapterManager;
using roadstar::common::util::Color;
using roadstar::common::util::ColorName;
using roadstar::localization::Localization;

struct CanvasTrans {
  CanvasTrans() = default;

  CanvasTrans(double offset_x, double offset_y, double view_w, double view_h,
              int w, int h, const Localization& loc,
              const common::sensor::SensorSource& dest_source =
                  common::sensor::VehicleCenter)
      : offset_x(offset_x),
        offset_y(offset_y),
        view_w(view_w),
        view_h(view_h),
        w(w),
        h(h) {
    scale_x = w / view_w;
    scale_y = h / view_h;
    world_imu_trans = common::SensorCoordinate::GetCoordTrans(
        dest_source, common::sensor::World, loc);
  }

  cv::Point TransUTM(double utm_x, double utm_y) const;
  cv::Point TransEgo(double ego_x, double ego_y) const;

  inline cv::Point Center() const {
    return cv::Point(w / 2, h / 2 - offset_y * scale_x);
  }

  common::CoordTransD world_imu_trans;
  int offset_x = 0;
  int offset_y = 0;
  double scale_x = 0;
  double scale_y = 0;
  double view_w = 0;
  double view_h = 0;
  int w = 0;
  int h = 0;
  std::string show_type;
  size_t seq_num = 0;
};

struct GridRenderer {
  explicit GridRenderer(GridRendererParam param = GridRendererParam());
  ~GridRenderer() = default;

  std::string Name() const {
    return "GridRenderer";
  }

  /*
   * If not specified, all the following numbers about grids are measured in
   * meters
   */
  double grid_cell_height_;
  double grid_cell_width_;
  // the frequency of showing grid lines in bold
  double grid_bold_gap_;
  // the thickness (measured in opencv fashion) of grid lines
  double grid_line_width_;
  cv::Scalar grid_line_color_;

  void DrawGridLines(const CanvasTrans& trans, cv::Mat* mat);
};

class MapRenderer {
 public:
  explicit MapRenderer(MapRendererParam param = MapRendererParam());
  ~MapRenderer() = default;

  void RenderMap(const Localization& loc, const CanvasTrans& trans,
                 cv::Mat* map_img,
                 const cv::Scalar& bg_color = 0xffffff_rgb .cv_rgb());

  std::string Name() const {
    return "MapRenderer";
  }

 private:
  double hdmap_radius_;
  double lane_marker_width_;
  cv::Scalar lane_marker_color_;
  double vehicle_length_;
  double vehicle_width_;
  cv::Scalar vehicle_color_;

  /*
   * Whether to draw grid lines on the image, for the convenience of
   * estimating distance by eye.
   * If not specified, all the following numbers about grids are measured in
   * meters
   */
  bool draw_grid_lines_ = false;

  void DrawMap(const Localization& loc, const CanvasTrans& trans, cv::Mat* mat);

  void DrawLaneMarker(const roadstar::hdmap::Lanemarker& lane_marker,
                      double offset_x, double offset_y,
                      const CanvasTrans& trans, cv::Mat* mat);

  void DrawConnection(const roadstar::hdmap::Connection& connection,
                      double offset_x, double offset_y,
                      const CanvasTrans& trans, cv::Mat* mat);

  HDMapInput* hdmap_input_;
  GridRenderer grid_renderer_;
};

class CvRenderer {
 public:
  void Render(const std::string& window_name, const cv::Mat& img) {
    std::lock_guard<std::mutex> cv_lock(cv_mutex_);
    cv::imshow(window_name, img);
    cv::waitKey(1);
  }

 private:
  std::mutex cv_mutex_;
  DECLARE_SINGLETON(CvRenderer);
};

void GetObjectRenderRect(double center_x, double center_y, double length,
                         double width, double theta, const CanvasTrans& trans,
                         std::vector<cv::Point>* rect);

void DrawRect(const std::vector<cv::Point>& rect, const cv::Scalar& color,
              cv::Mat* mat);

std::vector<Eigen::Vector2d> GetCornerPoints(const Eigen::Vector2d& center,
                                             const double length,
                                             const double width,
                                             const double theta);

}  // namespace visualization
}  // namespace common
}  // namespace roadstar

#endif  // MODULES_COMMON_VISUALIZATION_UTIL_H_
