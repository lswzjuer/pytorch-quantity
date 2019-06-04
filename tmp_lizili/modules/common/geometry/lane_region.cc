#include "modules/common/geometry/lane_region.h"

#include <iostream>

#include "modules/common/log.h"

namespace roadstar {
namespace common {
namespace geometry {

namespace {
constexpr double kResolution = 0.1;
constexpr int kSizeX = 1500;
constexpr int kSizeY = 1000;
const int kSize = kSizeY * kSizeX;
constexpr int kCenterX = 1000;
constexpr int kCenterY = 500;
}  // namespace

LanePoint::LanePoint(const int x, const int y, const double theta)
    : x(x), y(y), theta(theta) {}

LaneRegion::LaneRegion() : x0_(0), y0_(0), theta0_(0) {}

LaneRegion::LaneRegion(double x0, double y0, double theta0)
    : x0_(x0), y0_(y0), theta0_(theta0) {}

void LaneRegion::LabelLaneRegion(const Curve &left_curve,
                                 const Curve &right_curve, const int &size_x,
                                 const int &size_y, const int &center_x,
                                 const int &center_y,
                                 std::vector<unsigned char> *map,
                                 unsigned char index) {
  DetectLanePoints(left_curve, size_x, size_y, center_x, center_y,
                   &left_lane_points_);
  DetectLanePoints(right_curve, size_x, size_y, center_x, center_y,
                   &right_lane_points_);
  if (left_lane_points_.size() == 0 || right_lane_points_.size() == 0) {
    std::fill(map->begin(), map->end(), 255);
    AERROR << "Lane points empty";
    return;
  }
  LanePoint left_start_point = left_lane_points_.front();
  LanePoint left_end_point = left_lane_points_.back();
  LanePoint right_start_point = right_lane_points_.front();
  LanePoint right_end_point = right_lane_points_.back();
  DetectLineBoundary(left_start_point, right_start_point, &down_boundary_);
  DetectLineBoundary(left_end_point, right_end_point, &up_boundary_);
  DetectBoundaryPairs();
  DetectLaneRegionOnPoints(size_x, size_y, map, index);
}

void LaneRegion::LabelLaneRegion(
    const std::vector<roadstar::common::Polygon> &poly_points_list,
    const int &size_x, const int &size_y, const int &center_x,
    const int &center_y, std::vector<unsigned char> *map, unsigned char index) {
  unsigned char label = index | index;
  cv::Mat grid_mat(size_y, size_x, CV_8U, static_cast<void *>(map->data()));
  if (poly_points_list.size() == 0) {
    AERROR_EVERY(100) << "poly_points_list size is 0   ";
    return;
  }
  for (const auto &poly_points : poly_points_list) {
    std::vector<cv::Point> root_points;
    if (poly_points.points_size() == 0) {
      AERROR << "poly_points size is 0   ";
      return;
    }
    for (const auto &point : poly_points.points()) {
      cv::Point cv_point;
      TransPoseToGrid(center_x, center_y, point.x(), point.y(), &(cv_point.x),
                      &(cv_point.y));
      root_points.push_back(cv_point);
    }
    const cv::Point *ppt[1] = {
        static_cast<const cv::Point *>(root_points.data())};
    int npt[] = {static_cast<int>(root_points.size())};
    cv::fillPoly(grid_mat, ppt, npt, 1, cv::Scalar(label));
  }
}

void LaneRegion::LabelLaneRegion(
    const std::vector<std::vector<std::tuple<double, double>>>
        &poly_points_list,
    const int &size_x, const int &size_y, const int &center_x,
    const int &center_y, std::vector<unsigned char> *map, unsigned char index) {
  unsigned char label = index | index;
  cv::Mat grid_mat(size_y, size_x, CV_8U, static_cast<void *>(map->data()));
  if (poly_points_list.size() == 0) {
    AERROR_EVERY(100) << "poly_points_list size is 0   ";
    return;
  }
  for (const auto &poly_points : poly_points_list) {
    std::vector<cv::Point> root_points;
    if (poly_points.size() == 0) {
      AERROR << "poly_points size is 0   ";
      return;
    }
    for (const auto &point_tuple : poly_points) {
      cv::Point cv_point;
      TransPoseToGrid(center_x, center_y, std::get<0>(point_tuple),
                      std::get<1>(point_tuple), &(cv_point.x), &(cv_point.y));
      root_points.push_back(cv_point);
    }
    const cv::Point *ppt[1] = {
        static_cast<const cv::Point *>(root_points.data())};
    int npt[] = {static_cast<int>(root_points.size())};
    cv::fillPoly(grid_mat, ppt, npt, 1, cv::Scalar(label));
  }
}

void LaneRegion::LabelLaneRegion(const Curve &left_curve,
                                 const Curve &right_curve,
                                 std::vector<unsigned char> *map,
                                 unsigned char index) {
  LabelLaneRegion(left_curve, right_curve, kSizeX, kSizeY, kCenterX, kCenterY,
                  map, index);
}

void LaneRegion::LabelLaneRegion(
    const std::vector<std::vector<std::tuple<double, double>>>
        &poly_points_list,
    std::vector<unsigned char> *map, unsigned char index) {
  LabelLaneRegion(poly_points_list, kSizeX, kSizeY, kCenterX, kCenterY, map,
                  index);
}

void LaneRegion::DetectLanePoints(const Curve &curve, const int &size_x,
                                  const int &size_y, const int &center_x,
                                  const int &center_y,
                                  std::vector<LanePoint> *lane_points) {
  lane_points->clear();
  std::vector<double> s_list;
  double step = kResolution * 0.5;
  int s_num = std::floor(curve.length() / step);
  s_list.reserve(s_num);
  for (int i = 0; i < s_num; i++) {
    s_list.push_back(i * step);
  }
  std::vector<double> x_list, y_list, theta_list;
  curve.x(s_list, &x_list);
  curve.y(s_list, &y_list);
  curve.theta(s_list, &theta_list);
  bool pre_in_map = false;
  for (int i = 0; i < s_num; i++) {
    double x = x_list.at(i);
    double y = y_list.at(i);
    double theta = theta_list.at(i);
    int grid_x = -1, grid_y = -1;
    TransPoseToGrid(center_x, center_y, x, y, &grid_x, &grid_y);
    bool in_map = false;
    if (grid_x >= 0 && grid_x < size_x && grid_y >= 0 && grid_y < size_y) {
      in_map = true;
    }
    if (in_map) {
      lane_points->emplace_back(grid_x, grid_y, theta);
    } else if (pre_in_map) {
      break;
    }
    pre_in_map = in_map;
  }
}
void LaneRegion::DetectLanePoints(const Curve &curve,
                                  std::vector<LanePoint> *lane_points) {
  DetectLanePoints(curve, kSizeX, kSizeY, kCenterX, kCenterY, lane_points);
}

void LaneRegion::DetectLineBoundary(const LanePoint &start_point,
                                    const LanePoint &end_point,
                                    std::vector<LanePoint> *lane_points) {
  const int x_diff = end_point.x - start_point.x,
            y_diff = end_point.y - start_point.y;
  if (x_diff == 0 && y_diff == 0) {
    AERROR << "start point and end point are the same!";
    return;
  }
  float theta = 0;
  if (x_diff == 0) {
    theta = M_PI / 2;
  } else {
    theta = atan(y_diff / x_diff);
  }

  if (fabs(x_diff) > fabs(y_diff)) {
    const float k =
        y_diff / x_diff;  // NOLINT TODO(yujincheng): might divided by zero
    const float b = end_point.y - k * end_point.x;
    const int sign = start_point.x < end_point.x ? 1 : -1;
    for (int x = start_point.x; x <= end_point.x; x += sign) {
      int y = static_cast<int>(x * k + b);
      lane_points->emplace_back(x, y, theta);
    }
  } else {
    const float k = x_diff / y_diff;
    const float b = end_point.x - k * end_point.y;
    const int sign = start_point.y < end_point.y ? 1 : -1;
    for (int y = start_point.y; y < end_point.y; y += sign) {
      const int x = static_cast<int>(y * k + b);
      lane_points->emplace_back(x, y, theta);
    }
  }
}

void LaneRegion::DetectBoundaryPairs() {
  boundary_pairs_.clear();
  if (left_lane_points_.size() > 0) {
    for (const auto &point : left_lane_points_) {
      const auto it = boundary_pairs_.find(point.x);
      if (it != boundary_pairs_.end()) {
        if (point.y > it->second.second) it->second.second = point.y;
        if (point.y < it->second.first) it->second.first = point.y;
      } else {
        boundary_pairs_[point.x] = std::pair<int, int>(point.y, point.y);
      }
    }
  }

  if (right_lane_points_.size() > 0) {
    for (const auto &point : right_lane_points_) {
      const auto it = boundary_pairs_.find(point.x);
      if (it != boundary_pairs_.end()) {
        if (point.y > it->second.second) it->second.second = point.y;
        if (point.y < it->second.first) it->second.first = point.y;
      } else {
        boundary_pairs_[point.x] = std::pair<int, int>(point.y, point.y);
      }
    }
  }

  if (up_boundary_.size() > 0) {
    for (const auto &point : up_boundary_) {
      const auto it = boundary_pairs_.find(point.x);
      if (it != boundary_pairs_.end()) {
        if (point.y > it->second.second) it->second.second = point.y;
        if (point.y < it->second.first) it->second.first = point.y;
      } else {
        boundary_pairs_[point.x] = std::pair<int, int>(point.y, point.y);
      }
    }
  }

  if (down_boundary_.size() > 0) {
    for (const auto &point : down_boundary_) {
      const auto it = boundary_pairs_.find(point.x);
      if (it != boundary_pairs_.end()) {
        if (point.y > it->second.second) it->second.second = point.y;
        if (point.y < it->second.first) it->second.first = point.y;
      } else {
        boundary_pairs_[point.x] = std::pair<int, int>(point.y, point.y);
      }
    }
  }
}

void LaneRegion::DetectLaneRegionOnPoints(const int &size_x, const int &size_y,
                                          std::vector<unsigned char> *grid_map,
                                          unsigned char label) {
  int size = size_x * size_y;
  grid_map->resize(size);
  if (boundary_pairs_.size() > 0) {
    for (const auto &pair : boundary_pairs_) {
      const int x = pair.first;
      const int small_y = pair.second.first;
      const int large_y = pair.second.second;
      const int small_index = small_y * size_x + x;
      const int large_index = large_y * size_x + x;
      for (int index = small_index; index <= large_index; index += size_x) {
        if (index < size && index > 0) {
          grid_map->at(index) = grid_map->at(index) | label;
        }
      }
    }
  }
}

// grid_map should be initialized before this function;
void LaneRegion::DetectLaneRegionOnPoints(std::vector<unsigned char> *grid_map,
                                          unsigned char label) {
  DetectLaneRegionOnPoints(kSizeX, kSizeY, grid_map, label);
}

inline void LaneRegion::TransPoseToGrid(int center_x, int center_y, double x,
                                        double y, int *grid_x, int *grid_y) {
  double x_t = std::cos(theta0_) * (x - x0_) + std::sin(theta0_) * (y - y0_);
  double y_t = -std::sin(theta0_) * (x - x0_) + std::cos(theta0_) * (y - y0_);
  *grid_x = center_x - x_t / kResolution;
  *grid_y = center_y - y_t / kResolution;
}

inline void LaneRegion::TransPoseToGrid(double x, double y, int *grid_x,
                                        int *grid_y) {
  TransPoseToGrid(kCenterX, kCenterY, x, y, grid_x, grid_y);
}

}  // namespace geometry
}  // namespace common
}  // namespace roadstar
