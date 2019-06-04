#ifndef MODULES_PERCEPTION_TEST_PERCEPTION_OBSTACLE_MATCH_OBSTACLE_MATCH_INTERFACE_H
#define MODULES_PERCEPTION_TEST_PERCEPTION_OBSTACLE_MATCH_OBSTACLE_MATCH_INTERFACE_H

#include <map>
#include <memory>
#include "modules/integration_test/perception/obstacle/model/label_frame_model.h"
#include "modules/integration_test/perception/obstacle/model/perception_frame_model.h"
#include "modules/integration_test/perception/obstacle/model/report_model.h"

namespace roadstar {
namespace integration_test {

class ObstacleMatchInterface {
 public:
  virtual std::shared_ptr<ReportModel> Match(
      const std::map<int, LabelFrameModel> &labeled_obstacles,
      const std::map<int, PerceptionFrameModel> &perception_obstacles) = 0;
  virtual void ComputeLabeledObstacleVelocity(
      const std::map<int, LabelFrameModel> &labeled_frames,
      const std::map<int, PerceptionFrameModel> &perception_frames) = 0;
  virtual ~ObstacleMatchInterface() = default;
};

}  // namespace integration_test
}  // namespace roadstar

#endif
