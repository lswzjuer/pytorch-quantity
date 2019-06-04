#include "modules/integration_test/perception/obstacle/match/match_by_obstacle_overlap.h"

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "modules/common/log.h"
#include "modules/integration_test/perception/util/tool.h"

namespace roadstar {
namespace integration_test {

namespace {
constexpr double MATCH_STANDARD_HIGHWAY = 0.5;
constexpr double MATCH_STANDARD_CITY = 0.3;
};

using MatchObstacle = std::pair<std::shared_ptr<LabeledObstacleModel>,
                                std::shared_ptr<PerceptionObstacleModel>>;

int MatchByObstacleOverlap::ComparePerFrame(
    int frame, const LabelFrameModel& label_frame_model,
    const PerceptionFrameModel& perception_frame_model,
    ReportModel* report_model, std::vector<double>* velocity_sim_this_frame,
    std::vector<double>* velocity_diff_norm_this_frame) {
  int match_count = 0;
  size_t size = perception_frame_model.Size();
  LocationModel* location_model =
      const_cast<PerceptionFrameModel*>(&perception_frame_model)
          ->GetLocationModel();
  std::set<int> match_set;
  for (size_t i = 0; i < size; ++i) {
    PerceptionObstacleModel* perception_ego_front_obstacle_model =
        const_cast<PerceptionFrameModel*>(&perception_frame_model)
            ->GetEgoFrontTypeModelAt(i);
    PerceptionObstacleModel* perception_utm_obstacle_model =
        const_cast<PerceptionFrameModel*>(&perception_frame_model)
            ->GetUtmTypeModelAt(i);
    for (size_t j = 0; j < label_frame_model.Size(); ++j) {
      if (match_set.count(j) == 1) {
        continue;
      }
      LabeledObstacleModel* labeled_ego_front_obstacle_model =
          const_cast<LabelFrameModel*>(&label_frame_model)
              ->GetEgoFrontTypeModelAt(j);
      LabeledObstacleModel* labeled_velodyne_obstacle_model =
          const_cast<LabelFrameModel*>(&label_frame_model)
              ->GetVelodyneTypeModelAt(j);
      if (IsMatch(frame, i, *perception_utm_obstacle_model, *location_model,
                  *labeled_velodyne_obstacle_model)) {
        SaveMatchObstacle(frame, *labeled_ego_front_obstacle_model,
                          *perception_ego_front_obstacle_model, report_model);
        match_count++;
        match_set.insert(j);

        double similarity = CalculateVelocitySimilarity(
            labeled_ego_front_obstacle_model->GetVelocity(),
            perception_ego_front_obstacle_model->GetVelocity(),
            labeled_ego_front_obstacle_model->GetHeading(),
            perception_ego_front_obstacle_model->GetHeading());
        (*velocity_sim_this_frame).push_back(similarity);
        double diff_norm = CalculateVelocityDiffNorm(
            labeled_ego_front_obstacle_model->GetVelocity(),
            perception_ego_front_obstacle_model->GetVelocity(),
            labeled_ego_front_obstacle_model->GetHeading(),
            perception_ego_front_obstacle_model->GetHeading());
        (*velocity_diff_norm_this_frame).push_back(diff_norm);
        break;
      }
    }
  }
  MatchByObstacleOverlap::ShowMatachRes(perception_frame_model.Size(),
                                        label_frame_model.Size(), match_count,
                                        frame);
  return match_count;
}

bool MatchByObstacleOverlap::IsMatch(
    const int frame, const int index,
    const PerceptionObstacleModel& perception_obstacle,
    const LocationModel& location_model,
    const LabeledObstacleModel& labeled_obstacle) {
  std::string version = config_->GetValueViaKey("perception");
  std::vector<PointENU> perception_ground =
      perception_obstacle.GetEgoFrontTypePolygon(location_model, version);
  polygon_t perception_pgn;
  FillPolygon(perception_ground, &perception_pgn);
  std::vector<PointENU> labeld_ground =
      labeled_obstacle.GetEgoFrontTypePolygon(version);
  polygon_t labeled_pgn;
  FillPolygon(labeld_ground, &labeled_pgn);
  std::vector<polygon_t> in, un;
  bg::intersection(labeled_pgn, perception_pgn, in);
  bg::union_(labeled_pgn, perception_pgn, un);
  if (in.empty()) {
    return false;
  }
  double inter_area = in.empty() ? 0 : bg::area(in.front());
  double union_area = bg::area(un.front());
  double overlap = inter_area / union_area;
  std::string dirve_scene = getenv("DRIVE_SCENE");
  double match_standard = 0;
  if (dirve_scene == "city") {
    match_standard = MATCH_STANDARD_CITY;
  } else {
    match_standard = MATCH_STANDARD_HIGHWAY;
  }
  if (overlap > match_standard) {
    // AINFO << std::setprecision(6) << "overlap = " << overlap;
    return true;
  }
  return false;
}

void MatchByObstacleOverlap::FillPolygon(const std::vector<PointENU>& pts,
                                         polygon_t* polygon) {
  for (auto& it : pts) {
    bg::append(polygon->outer(), point_t(it.x(), it.y()));
  }
}

void MatchByObstacleOverlap::ShowMatachRes(std::size_t perception_obs,
                                           std::size_t labeled_obs, int matches,
                                           int frame) {
  if (perception_obs > 0 && labeled_obs > 0) {
    AINFO << "frame = " << frame
          << " recall_average = " << matches / static_cast<double>(labeled_obs)
          << " percision_average = "
          << matches / static_cast<double>(perception_obs)
          << " match_count = " << matches
          << " labeled obstacles = " << labeled_obs
          << " perception obstacles = " << perception_obs;
  } else {
    AINFO << "frame = " << frame << " match_count = " << matches
          << " labeled obstacles = " << labeled_obs
          << " perception obstacles = " << perception_obs;
  }
}

void MatchByObstacleOverlap::SaveMatchObstacle(
    int frame, const LabeledObstacleModel& labeled_ob,
    const PerceptionObstacleModel& perception_ob, ReportModel* report) {
  std::shared_ptr<LabeledObstacleModel> match_labeled_ob =
      std::make_shared<LabeledObstacleModel>(labeled_ob);
  std::shared_ptr<PerceptionObstacleModel> match_perception_ob =
      std::make_shared<PerceptionObstacleModel>(perception_ob);
  report->AddMatchObstacle(
      std::make_pair(match_labeled_ob, match_perception_ob), frame);
}

}  // namespace integration_test
}  // namespace roadstar
