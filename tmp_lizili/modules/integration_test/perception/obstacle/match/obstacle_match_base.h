#ifndef MODULES_PERCEPTION_TEST_PERCEPTION_OBSTACLE_MATCH_OBSTACLE_MATCH_BASE_H
#define MODULES_PERCEPTION_TEST_PERCEPTION_OBSTACLE_MATCH_OBSTACLE_MATCH_BASE_H

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "modules/integration_test/perception/common/model/config_model.h"
#include "modules/integration_test/perception/obstacle/match/obstacle_match_interface.h"
#include "modules/integration_test/perception/obstacle/model/labeled_obstacle_model.h"
#include "modules/integration_test/perception/obstacle/model/location_model.h"
#include "modules/integration_test/perception/obstacle/model/perception_obstacle_model.h"
#include "modules/integration_test/perception/util/tool.h"

namespace roadstar {
namespace integration_test {

class ObstacleMatchBase : public ObstacleMatchInterface {
 public:
  explicit ObstacleMatchBase(const std::shared_ptr<ConfigModel>& config)
      : config_(config) {}

  std::shared_ptr<ReportModel> Match(
      const std::map<int, LabelFrameModel>& labeled_obstacles,
      const std::map<int, PerceptionFrameModel>& perception_obstacles) override;

  virtual int ComparePerFrame(
      int frame, const LabelFrameModel& label_frame_model,
      const PerceptionFrameModel& perception_frame_model,
      ReportModel* report_model, std::vector<double>* velocity_sim_this_frame,
      std::vector<double>* velocity_diff_norm_this_frame) = 0;

  const std::map<int, LabelFrameModel>& GetLabeledObstacles() {
    return labeled_obstacles_;
  }
  const std::map<int, PerceptionFrameModel>& GetPerceptionObstacles() {
    return perception_obstacles_;
  }
  void ComputeLabeledObstacleVelocity(
      const std::map<int, LabelFrameModel>& labeled_frames,
      const std::map<int, PerceptionFrameModel>& perception_frames) override;

  void CalculateLabeledObstacleVelocity(
      std::map<int, PerceptionFrameModel>::const_iterator prev_perception_it,
      std::map<int, LabelFrameModel>::const_iterator prev_label_it,
      std::map<int, PerceptionFrameModel>::const_iterator next_perception_it,
      std::map<int, LabelFrameModel>::const_iterator next_label_it,
      std::map<int, LabelFrameModel>::iterator current_label_it,
      size_t obstacle_id);

  double CalculateVelocitySimilarity(double v1, double v2, double heading1,
                                     double heading2);

  double CalculateVelocityDiffNorm(double v1, double v2, double heading1,
                                   double heading2);

 private:
  void FilterLabelObstacle(
      const std::map<int, LabelFrameModel>& labeled_obstacles);
  void FilterPerceptionObstacle(
      const std::map<int, PerceptionFrameModel>& perception_obstacles);
  void OutputLostPerceptionFramesInfo();
  LocationModel* GetLocationModelAt(int frame);

  template <typename T>
  T GetPn(const std::vector<T>& values, unsigned int n);
  void SaveFrameMatchRes(const int& frame, const int& match_count,
                         const size_t& perception_obstacles_size,
                         const size_t& label_obstacles_size,
                         ReportModel* report_model);

 protected:
  std::shared_ptr<ConfigModel> config_;
  std::map<int, LabelFrameModel> labeled_obstacles_;
  std::map<int, PerceptionFrameModel> perception_obstacles_;
  Tool tool_;
};

}  // namespace integration_test
}  // namespace roadstar

#endif
