#ifndef MODULES_INTEGRATION_TEST_PERCEPTION_OBSTACLE_MODEL_REPORT_MODEL_H
#define MODULES_INTEGRATION_TEST_PERCEPTION_OBSTACLE_MODEL_REPORT_MODEL_H

#include <memory>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "modules/integration_test/perception/obstacle/model/labeled_obstacle_model.h"
#include "modules/integration_test/perception/obstacle/model/perception_obstacle_model.h"

namespace roadstar {
namespace integration_test {

class ReportModel {
 public:
  using MatchObstacle = std::pair<std::shared_ptr<LabeledObstacleModel>,
                                  std::shared_ptr<PerceptionObstacleModel>>;

  void AddMatchObstacle(const MatchObstacle& model, int frame);
  void SetTotalPerceptionFrames(int frames);
  void SetPrecisionAverage(double precision);
  void SetRecallAverage(double recall);
  void SetTotalLabelFrames(int frames);
  void SetTotalMatchFrames(int frames);
  void SetZeroObstacleFrames(int frames);
  void SetTotalObstaclesOfPerceptionForFrame(int frame, int count);
  void SetTotalObstaclesOfLabelForFrame(int frame, int count);
  void SetMatchSizeForFrame(int frame, int count);
  void SetRecallPercentForFrame(int frame, double percent);
  void SetPrecisionPercentForFrame(int frame, double percent);
  void SetDriveScene(const std::string& scene);

  json ToJson();
  void FromJson(const json& value);

  bool ParseFromFile(const std::string& file);
  bool SerializeToFile(const std::string& file);

  void Diff(ReportModel* others, const std::string& file);

  void SetVelocityP50Global(double p50);
  void SetVelocityP95Global(double p95);
  void SetVelocityP50ForFrame(int frame, double p50);
  void SetVelocityP95ForFrame(int frame, double p95);
  void SetVelocityDiffNormAverage(double value);

  json MatchFramesToJson();
  void MatchFramesFromJson(const json& match_frames);
  template <class T>
  int Compare(T value1, T value2) {
    if (value1 > value2) {
      return 1;
    } else if (value1 < value2) {
      return -1;
    } else {
      return 0;
    }
  }
  template <class T>
  std::string Format(T value1, T value2) {
    std::string str_operator;
    if (value1 > value2) {
      str_operator = " > ";
    } else if (value1 < value2) {
      str_operator = " < ";
    } else {
      str_operator = " == ";
    }
    return "self : " + std::to_string(value1) + str_operator + " others : " +
           std::to_string(value2);
  }

 private:
  std::map<int, std::vector<MatchObstacle>> matches_;
  std::map<int, int> match_size_per_frame_;
  std::map<int, int> label_obs_per_frame_;
  std::map<int, int> perception_obs_per_frame_;
  std::map<int, double> recall_percent_per_frame_;
  std::map<int, double> precision_percent_per_frame_;
  int total_perception_frames_;
  double precision_average_;
  double recall_average_;
  int zero_ob_frames_;
  int total_labeled_frames_;
  int total_match_frames_;
  double velocity_sim_p50_global_;
  double velocity_sim_p95_global_;
  double velocity_diff_norm_average_;
  std::string dirve_scene_;
  std::map<int, double> velocity_sim_p50_per_frame_;
  std::map<int, double> velocity_sim_p95_per_frame_;
};

}  // namespace integration_test
}  // namespace roadstar

#endif
