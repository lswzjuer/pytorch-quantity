#ifndef MODULES_INTEGRATION_TEST_PERCEPTION_TRAFFIC_LIGHT_MODEL_TRAFFIC_LIGHT_DETECTION_H_
#define MODULES_INTEGRATION_TEST_PERCEPTION_TRAFFIC_LIGHT_MODEL_TRAFFIC_LIGHT_DETECTION_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "modules/integration_test/perception/traffic_light/model/traffic_light_model.h"

namespace roadstar {
namespace integration_test {

typedef std::shared_ptr<TrafficLightModel> TrafficLightPtr;
typedef TrafficLightPtr LabeledTrafficLightPtr;
typedef TrafficLightPtr PerceptionTrafficLightPtr;
typedef std::pair<LabeledTrafficLightPtr, PerceptionTrafficLightPtr>
    TrafficLightPtrPair;

class TrafficLightFrameModel {
  friend class TrafficLightDetectionReportModel;

 public:
  TrafficLightFrameModel* SetTimestamp(double timestamp);
  TrafficLightFrameModel* SetRecall(double recall);
  TrafficLightFrameModel* SetPrecision(double percision);
  TrafficLightFrameModel* SetMatch(int match);
  TrafficLightFrameModel* SetLabeledTotal(int total);
  TrafficLightFrameModel* SetPerceptionTotal(int total);
  TrafficLightFrameModel* AddTrafficLightPair(const TrafficLightPtrPair& pair);
  TrafficLightFrameModel* SetPerceptionCountdownLightsSize(const int& size);
  TrafficLightFrameModel* SetLabeledCountdownLightsSize(const int& size);
  json ToJson();

  TrafficLightFrameModel* Calculate();
  TrafficLightFrameModel* CalculateWithoutIgnore();
  TrafficLightFrameModel* ShowInfo();
  double GetTimestamp() const;
  uint32_t Size() const;
  uint32_t IgonreTrafficLightSize() const;

 private:
  std::vector<TrafficLightPtrPair> traffic_light_pairs_;
  int perception_total_ = 0;
  int labeled_total_ = 0;
  int match_ = 0;
  int countdown_light_match_ = 0;
  int perception_countdown_lights_ = 0;
  int labeled_countdown_lights_ = 0;
  double timestamp_ = 0;
  double recall_ = 0;     // :matches/labeled
  double precision_ = 0;  // :matches/perception
};

typedef std::shared_ptr<TrafficLightFrameModel> TrafficLightsFramePtr;

class TrafficLightDetectionReportModel {
 public:
  TrafficLightDetectionReportModel* SetTotalPerceptionFrames(int total);
  TrafficLightDetectionReportModel* SetPrecisionAverage(
      double precision_average);
  TrafficLightDetectionReportModel* SetRecallAverage(double recall_average);
  TrafficLightDetectionReportModel* SetTotalLabeledFrames(int total);
  TrafficLightDetectionReportModel* SetTotalMatchFrames(int total);
  TrafficLightDetectionReportModel* SetWeather(std::string weather);
  TrafficLightDetectionReportModel* AddTrafficLightFramePtr(
      const TrafficLightsFramePtr&);

  json ToJson();
  // void FromJson(const json& value);

  // bool ParseFromFile(const std::string& file);
  bool SerializeToFile(const std::string& file);
  TrafficLightDetectionReportModel* Calculate();
  TrafficLightDetectionReportModel* ShowInfo();

 private:
  std::vector<TrafficLightsFramePtr> matches_;
  int perception_traffic_lights_total_ = 0;
  int labeled_traffic_lights_total_ = 0;
  int match_traffic_lights_ = 0;
  int total_perception_frames_ = 0;
  double precision_average_ = 0;
  double recall_average_ = 0;
  double countdown_time_recall_ = 0;
  double countdown_time_precision_ = 0;
  int total_labeled_frames_ = 0;
  int total_match_frames_ = 0;
  std::string weather_ = "clearness";
};

typedef std::shared_ptr<TrafficLightDetectionReportModel>
    TrafficLightDetectionReportModelPtr;

}  // namespace integration_test
}  // namespace roadstar

#endif
