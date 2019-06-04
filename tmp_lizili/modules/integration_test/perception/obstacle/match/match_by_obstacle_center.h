#ifndef MODULES_PERCEPTION_TEST_PERCEPTION_OBSTACLE_MATCH_MATCH_BY_OBSTACLE_CENTER_H
#define MODULES_PERCEPTION_TEST_PERCEPTION_OBSTACLE_MATCH_MATCH_BY_OBSTACLE_CENTER_H

#include <iostream>
#include <memory>
#include <vector>

#include "modules/integration_test/perception/common/model/config_model.h"
#include "modules/integration_test/perception/obstacle/match/obstacle_match_base.h"

namespace roadstar {
namespace integration_test {

class MatchByObstacleCenter : public ObstacleMatchBase {
 public:
  explicit MatchByObstacleCenter(const std::shared_ptr<ConfigModel>& config)
      : ObstacleMatchBase(config) {}

  const double DISTANCE = pow(2.5, 2) + pow(1, 2);

  int ComparePerFrame(int frame, const LabelFrameModel& label_frame_model,
                      const PerceptionFrameModel& perception_frame_model,
                      ReportModel* report_model,
                      std::vector<double>* velocity_sim_this_frame,
                      std::vector<double>* velocity_diff_norm_this_frame) override;
};

}  // namespace integration_test
}  // namespace roadstar

#endif
