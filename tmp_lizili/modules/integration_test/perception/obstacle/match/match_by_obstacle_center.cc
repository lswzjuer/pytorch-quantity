#include "modules/integration_test/perception/obstacle/match/match_by_obstacle_center.h"

#include <memory>
#include <utility>
#include <vector>

#include "modules/common/log.h"
#include "modules/integration_test/perception/util/tool.h"

namespace roadstar {
namespace integration_test {

using MatchObstacle = std::pair<std::shared_ptr<LabeledObstacleModel>,
                                std::shared_ptr<PerceptionObstacleModel>>;

int MatchByObstacleCenter::ComparePerFrame(
    int frame, const LabelFrameModel& label_frame_model,
    const PerceptionFrameModel& perception_frame_model,
    ReportModel* report_model, std::vector<double>* velocity_sim_this_frame,
    std::vector<double>* velocity_diff_norm_this_frame) {
  int match_count = 0;
  size_t size = perception_frame_model.Size();
  for (size_t i = 0; i < size; ++i) {
    PerceptionObstacleModel* perception_obstacle_model =
        (const_cast<PerceptionFrameModel*>(&perception_frame_model))
            ->GetEgoFrontTypeModelAt(i);
    for (size_t j = 0; j < label_frame_model.Size(); ++j) {
      LabeledObstacleModel* labeled_obstacle_model =
          (const_cast<LabelFrameModel*>(&label_frame_model))
              ->GetEgoFrontTypeModelAt(j);
      double ab_x =
          perception_obstacle_model->GetX() - labeled_obstacle_model->GetX();
      double ab_y =
          perception_obstacle_model->GetY() - labeled_obstacle_model->GetY();
      if (pow(ab_x, 2.0) + pow(ab_y, 2.0) < DISTANCE) {
        std::shared_ptr<LabeledObstacleModel> match_labeled_ob =
            std::make_shared<LabeledObstacleModel>(*labeled_obstacle_model);
        std::shared_ptr<PerceptionObstacleModel> match_perception_ob =
            std::make_shared<PerceptionObstacleModel>(
                *perception_obstacle_model);
        report_model->AddMatchObstacle(
            std::make_pair(match_labeled_ob, match_perception_ob), frame);
        match_count++;
        break;
      }
    }
  }
  return match_count;
}

}  // namespace integration_test
}  // namespace roadstar
