#ifndef MODULES_PERCEPTION_TEST_PERCEPTION_OBSTACLE_MATCH_MATCH_BY_OBSTACLE_OVERLAP_H
#define MODULES_PERCEPTION_TEST_PERCEPTION_OBSTACLE_MATCH_MATCH_BY_OBSTACLE_OVERLAP_H

#include <iostream>
#include <memory>
#include <vector>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/geometries.hpp>

#include "modules/common/proto/geometry.pb.h"
#include "modules/integration_test/perception/common/model/config_model.h"
#include "modules/integration_test/perception/obstacle/match/obstacle_match_base.h"

namespace roadstar {
namespace integration_test {

namespace bg = boost::geometry;
using point_t = bg::model::point<double, 2, bg::cs::cartesian>;
using polygon_t = bg::model::polygon<point_t>;
using PointENU = roadstar::common::PointENU;  // actually the z value here isn't
                                              // useful for now.

class MatchByObstacleOverlap : public ObstacleMatchBase {
 public:
  explicit MatchByObstacleOverlap(const std::shared_ptr<ConfigModel>& config)
      : ObstacleMatchBase(config) {}

  int ComparePerFrame(
      int frame, const LabelFrameModel& label_frame_model,
      const PerceptionFrameModel& perception_frame_model,
      ReportModel* report_model, std::vector<double>* velocity_sim_this_frame,
      std::vector<double>* velocity_diff_norm_this_frame) override;

 private:
  bool IsMatch(const int frame, const int index,
               const PerceptionObstacleModel& perception_obstacle,
               const LocationModel& location_model,
               const LabeledObstacleModel& labeled_obstacle);
  void FillPolygon(const std::vector<PointENU>& pts, polygon_t* polygon);

  void ShowMatachRes(std::size_t perception_obs, std::size_t labeled_obs,
                     int matches, int frame);
  void SaveMatchObstacle(int frame, const LabeledObstacleModel& labeled_ob,
                         const PerceptionObstacleModel& perception_ob,
                         ReportModel* report);
};

}  // namespace integration_test
}  // namespace roadstar

#endif
