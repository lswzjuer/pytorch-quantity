#ifndef MODULES_COMMON_GEOMETRY_LANE_REGION_H
#define MODULES_COMMON_GEOMETRY_LANE_REGION_H

#include <memory.h>
#include <map>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>
#include <opencv2/opencv.hpp>

#include "modules/common/geometry/curve.h"
#include "modules/common/proto/geometry.pb.h"

namespace roadstar {
namespace common {
namespace geometry {
// Struct stores lane point information.
typedef struct LanePoint {
  LanePoint(const int x, const int y, const double theta);
  int x;
  int y;
  double theta;
} LanePoint;

class LaneRegion {
 public:
  LaneRegion();

  LaneRegion(double x0, double y0, double theta0);

  void LabelLaneRegion(const Curve& left_curve, const Curve& right_curve,
                       std::vector<unsigned char>* map,
                       unsigned char label = 1);
  void LabelLaneRegion(const Curve& left_curve, const Curve& right_curve,
                       const int& size_x, const int& size_y,
                       const int& center_x, const int& center_y,
                       std::vector<unsigned char>* map,
                       unsigned char label = 1);
  void LabelLaneRegion(
      const std::vector<std::vector<std::tuple<double, double>>>&
          poly_points_list,
      std::vector<unsigned char>* map, unsigned char label = 1);
  void LabelLaneRegion(
      const std::vector<std::vector<std::tuple<double, double>>>&
          poly_points_list,
      const int& size_x, const int& size_y, const int& center_x,
      const int& center_y, std::vector<unsigned char>* map,
      unsigned char label = 1);
  void LabelLaneRegion(
      const std::vector<roadstar::common::Polygon>& poly_points_list,
      const int& size_x, const int& size_y, const int& center_x,
      const int& center_y, std::vector<unsigned char>* map,
      unsigned char label = 1);

 private:
  void DetectLanePoints(const Curve& curve, const int& size_x,
                        const int& size_y, const int& center_x,
                        const int& center_y,
                        std::vector<LanePoint>* lane_points);

  void DetectLanePoints(const Curve& curve,
                        std::vector<LanePoint>* lane_points);
  void DetectLaneRegionOnPoints(const int& size_x, const int& size_y,
                                std::vector<unsigned char>* grid_map,
                                unsigned char label);
  void DetectLaneRegionOnPoints(std::vector<unsigned char>* grid_map,
                                unsigned char label);
  void DetectBoundaryPairs();

  void DetectLineBoundary(const LanePoint& start_point,
                          const LanePoint& end_point,
                          std::vector<LanePoint>* lane_points);

  inline void TransPoseToGrid(double x, double y, int* grid_x, int* grid_y);

  inline void TransPoseToGrid(int center_x, int center_y, double x, double y,
                              int* grid_x, int* grid_y);
  double x0_, y0_, theta0_;
  std::vector<LanePoint> left_lane_points_;
  std::vector<LanePoint> right_lane_points_;
  std::vector<LanePoint> up_boundary_;
  std::vector<LanePoint> down_boundary_;
  std::map<int, std::pair<int, int>> boundary_pairs_;
};  // class LaneRegion
using LaneRegionPtr = std::unique_ptr<LaneRegion>;
}  // namespace geometry
}  // namespace common
}  // namespace roadstar
#endif  // MODULES_COMMON_GEOMETRY_LANE_REGION_H
