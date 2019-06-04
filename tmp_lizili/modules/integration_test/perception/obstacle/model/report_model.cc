#include "modules/integration_test/perception/obstacle/model/report_model.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include "modules/common/log.h"

namespace roadstar {
namespace integration_test {

void ReportModel::AddMatchObstacle(const MatchObstacle& model, int frame) {
  if (matches_.find(frame) != matches_.end()) {
    matches_[frame].push_back(model);
  } else {
    std::vector<MatchObstacle> obs;
    obs.push_back(model);
    matches_[frame] = obs;
  }
}

void ReportModel::SetPrecisionAverage(double precision) {
  precision_average_ = precision;
}

void ReportModel::SetVelocityDiffNormAverage(double value) {
  velocity_diff_norm_average_ = value;
}

void ReportModel::SetDriveScene(const std::string& scene) {
  dirve_scene_ = scene;
}

void ReportModel::SetMatchSizeForFrame(int frame, int count) {
  match_size_per_frame_[frame] = count;
}

void ReportModel::SetRecallAverage(double recall) {
  recall_average_ = recall;
}

void ReportModel::SetVelocityP50Global(double p50) {
  velocity_sim_p50_global_ = p50;
}

void ReportModel::SetVelocityP95Global(double p95) {
  velocity_sim_p95_global_ = p95;
}

void ReportModel::SetTotalPerceptionFrames(int frames) {
  total_perception_frames_ = frames;
}

void ReportModel::SetRecallPercentForFrame(int frame, double percent) {
  recall_percent_per_frame_[frame] = percent;
}
void ReportModel::SetPrecisionPercentForFrame(int frame, double percent) {
  precision_percent_per_frame_[frame] = percent;
}

void ReportModel::SetVelocityP50ForFrame(int frame, double p50) {
  velocity_sim_p50_per_frame_[frame] = p50;
}

void ReportModel::SetVelocityP95ForFrame(int frame, double p95) {
  velocity_sim_p95_per_frame_[frame] = p95;
}

void ReportModel::SetTotalLabelFrames(int frames) {
  total_labeled_frames_ = frames;
}

void ReportModel::SetTotalMatchFrames(int frames) {
  total_match_frames_ = frames;
}
void ReportModel::SetZeroObstacleFrames(int frames) {
  zero_ob_frames_ = frames;
}

void ReportModel::SetTotalObstaclesOfPerceptionForFrame(int frame, int count) {
  perception_obs_per_frame_[frame] = count;
}

void ReportModel::SetTotalObstaclesOfLabelForFrame(int frame, int count) {
  label_obs_per_frame_[frame] = count;
}

json ReportModel::ToJson() {
  json report;
  report["frames"] = total_match_frames_;
  report["drive_scene"] = dirve_scene_;
  report["perception_frames"] = total_perception_frames_;
  report["label_frames"] = total_labeled_frames_;
  report["precision_average"] = precision_average_;
  report["recall_average"] = recall_average_;
  report["zero_ob_frame_count"] = zero_ob_frames_;
  report["match_frames"] = MatchFramesToJson();
  report["velocity_precision_p50"] = velocity_sim_p50_global_;
  report["velocity_precision_p95"] = velocity_sim_p95_global_;
  report["velocity_diff_norm_average"] = velocity_diff_norm_average_;
  return report;
}

void ReportModel::FromJson(const json& value) {
  total_match_frames_ = value["frames"];
  dirve_scene_ = value["drive_scene"];
  total_perception_frames_ = value["perception_frames"];
  total_labeled_frames_ = value["label_frames"];
  precision_average_ = value["precision_average"];
  recall_average_ = value["recall_average"];
  zero_ob_frames_ = value["zero_ob_frame_count"];
  json match_frames = value["match_frames"];
  velocity_sim_p50_global_ = value["velocity_precision_p50"];
  velocity_sim_p95_global_ = value["velocity_precision_p95"];
  velocity_diff_norm_average_ = value["velocity_diff_norm_average"];
  MatchFramesFromJson(match_frames);
}

json ReportModel::MatchFramesToJson() {
  json match_frames;
  for (auto& it : matches_) {
    json matches;
    for (auto& sub_it : it.second) {
      json one_match;
      one_match["label_ob"] = sub_it.first->ToJson();
      one_match["perception_ob"] = sub_it.second->ToJson();
      matches.push_back(one_match);
    }
    json frame;
    frame["matches"] = matches;
    frame["label_obs_size"] = label_obs_per_frame_[it.first];
    frame["match_size"] = match_size_per_frame_[it.first];
    frame["perception_obs_size"] = perception_obs_per_frame_[it.first];
    frame["recall_percent"] = recall_percent_per_frame_[it.first];
    frame["precision_percent"] = precision_percent_per_frame_[it.first];

    frame["velocity_p50"] = velocity_sim_p50_per_frame_[it.first];
    frame["velocity_p95"] = velocity_sim_p95_per_frame_[it.first];
    match_frames[std::to_string(it.first)] = frame;
  }
  return match_frames;
}

void ReportModel::MatchFramesFromJson(const json& match_frames) {
  for (auto it = match_frames.begin(); it != match_frames.end(); ++it) {
    int key = atoi(it.key().c_str());
    json frame = it.value();
    label_obs_per_frame_[key] = frame["label_obs_size"];
    match_size_per_frame_[key] = frame["match_size"];
    perception_obs_per_frame_[key] = frame["perception_obs_size"];
    recall_percent_per_frame_[key] = frame["recall_percent"];
    precision_percent_per_frame_[key] = frame["precision_percent"];

    velocity_sim_p50_per_frame_[key] = frame["velocity_p50"];
    velocity_sim_p95_per_frame_[key] = frame["velocity_p95"];
    json matches_json = frame["matches"];
    std::vector<MatchObstacle> match_obstacles;
    for (auto one_match = matches_json.begin(); one_match != matches_json.end();
         ++one_match) {
      json label_ob = (*one_match)["label_ob"];
      json perception_ob = (*one_match)["perception_ob"];
      std::shared_ptr<LabeledObstacleModel> labeled_obstacle_model =
          std::make_shared<LabeledObstacleModel>(label_ob);
      std::shared_ptr<PerceptionObstacleModel> perception_obstacle_model =
          std::make_shared<PerceptionObstacleModel>(perception_ob);
      match_obstacles.push_back(
          std::make_pair(labeled_obstacle_model, perception_obstacle_model));
    }
    matches_[key] = match_obstacles;
  }
}

bool ReportModel::ParseFromFile(const std::string& file) {
  std::ifstream in_file(file);
  if (!in_file.is_open()) {
    AERROR << "file open error. path is " << file;
    return false;
  }

  json value;
  in_file >> value;
  FromJson(value);
  in_file.close();
  return true;
}

bool ReportModel::SerializeToFile(const std::string& file) {
  std::ofstream o(file, std::ios_base::out | std::ios_base::trunc);
  if (!o.is_open()) {
    AERROR << "file open error. path is " << file;
    return false;
  }
  json value = ToJson();
  o << std::setw(4) << value << std::endl;
  o.close();
  return true;
}

void ReportModel::Diff(ReportModel* others, const std::string& file) {
  json res;
  for (auto it = matches_.begin(); it != matches_.end(); ++it) {
    int key = it->first;
    if (others->matches_.find(key) == others->matches_.end()) {
      continue;
    }
    std::cout << "key = " << key << std::endl;
    json frame;
    frame["match_size"] = Format<int>(match_size_per_frame_[key],
                                      others->match_size_per_frame_[key]);
    frame["label_ob_size"] = Format<int>(label_obs_per_frame_[key],
                                         others->label_obs_per_frame_[key]);
    frame["perception_ob_size"] = Format<int>(
        perception_obs_per_frame_[key], others->perception_obs_per_frame_[key]);
    frame["recall"] = Format<double>(recall_percent_per_frame_[key],
                                     others->recall_percent_per_frame_[key]);
    frame["precision"] =
        Format<double>(precision_percent_per_frame_[key],
                       others->precision_percent_per_frame_[key]);

    frame["velocity_p50"] =
        Format<double>(velocity_sim_p50_per_frame_[key],
                       others->velocity_sim_p50_per_frame_[key]);
    frame["velocity_p95"] =
        Format<double>(velocity_sim_p95_per_frame_[key],
                       others->velocity_sim_p95_per_frame_[key]);
    if (Compare<double>(perception_obs_per_frame_[key],
                        others->perception_obs_per_frame_[key]) == 0 &&
        Compare<double>(precision_percent_per_frame_[key],
                        others->precision_percent_per_frame_[key]) == 0) {
      continue;
    }
    res[std::to_string(key)] = frame;
  }
  std::ofstream o(file, std::ios_base::out | std::ios_base::trunc);
  if (!o.is_open()) {
    AERROR << "file open error. path is " << file;
    return;
  }
  o << std::setw(4) << res << std::endl;
  o.close();
}

}  // namespace integration_test
}  // namespace roadstar
