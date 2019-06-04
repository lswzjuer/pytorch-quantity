#include "modules/integration_test/perception/traffic_light/model/traffic_light_detection_report_model.h"
#include <fstream>
#include <iomanip>

#include "modules/common/log.h"

namespace roadstar {
namespace integration_test {

TrafficLightFrameModel* TrafficLightFrameModel::SetTimestamp(double timestamp) {
  timestamp_ = timestamp;
  return this;
}

TrafficLightFrameModel* TrafficLightFrameModel::SetRecall(double recall) {
  recall_ = recall;
  return this;
}

TrafficLightFrameModel* TrafficLightFrameModel::SetPrecision(double precision) {
  precision_ = precision;
  return this;
}

TrafficLightFrameModel* TrafficLightFrameModel::AddTrafficLightPair(
    const TrafficLightPtrPair& pair) {
  traffic_light_pairs_.push_back(pair);
  return this;
}

TrafficLightFrameModel* TrafficLightFrameModel::SetMatch(int match) {
  match_ = match;
  return this;
}

TrafficLightFrameModel* TrafficLightFrameModel::SetLabeledTotal(int total) {
  labeled_total_ = total;
  return this;
}

TrafficLightFrameModel* TrafficLightFrameModel::SetPerceptionTotal(int total) {
  perception_total_ = total;
  return this;
}

double TrafficLightFrameModel::GetTimestamp() const {
  return timestamp_;
}

uint32_t TrafficLightFrameModel::Size() const {
  return traffic_light_pairs_.size();
}

uint32_t TrafficLightFrameModel::IgonreTrafficLightSize() const {
  uint32_t size = 0;
  for (auto& it : traffic_light_pairs_) {
    if (it.first->IsIgnore()) {
      ++size;
    }
  }
  return size;
}

TrafficLightFrameModel* TrafficLightFrameModel::Calculate() {
  match_ = traffic_light_pairs_.size();
  if (perception_total_ == 0) {
    precision_ = 0;
  } else {
    precision_ = match_ / static_cast<double>(perception_total_);
  }
  if (labeled_total_ == 0) {
    recall_ = 0;
  } else {
    recall_ = match_ / static_cast<double>(labeled_total_);
  }
  return this;
}

TrafficLightFrameModel* TrafficLightFrameModel::CalculateWithoutIgnore() {
  uint32_t ignore = IgonreTrafficLightSize();
  // AINFO << "ignore size = " << ignore;
  match_ = traffic_light_pairs_.size() - ignore;
  perception_total_ -= ignore;
  labeled_total_ -= ignore;
  if (perception_total_ == 0) {
    precision_ = 0;
  } else {
    precision_ = match_ / static_cast<double>(perception_total_);
  }
  if (labeled_total_ == 0) {
    recall_ = 0;
  } else {
    recall_ = match_ / static_cast<double>(labeled_total_);
  }
  countdown_light_match_ = 0;
  for (auto& it : traffic_light_pairs_) {
    if (it.first->GetCountDownTime() == it.second->GetCountDownTime() &&
        it.first->GetCountDownTime() >= 0) {
      ++countdown_light_match_;
    }
  }
  return this;
}

TrafficLightFrameModel* TrafficLightFrameModel::ShowInfo() {
  AINFO << "perception_total = " << perception_total_
        << ". labeled_total = " << labeled_total_
        << ". precision = " << precision_ << ". recall = " << recall_
        << ". match = " << match_ << std::setprecision(14)
        << ". timestamp = " << timestamp_;
  return this;
}

TrafficLightFrameModel*
TrafficLightFrameModel::SetPerceptionCountdownLightsSize(const int& size) {
  perception_countdown_lights_ = size;
  return this;
}

TrafficLightFrameModel* TrafficLightFrameModel::SetLabeledCountdownLightsSize(
    const int& size) {
  labeled_countdown_lights_ = size;
  return this;
}

json TrafficLightFrameModel::ToJson() {
  json model;
  model["perception_total"] = perception_total_;
  model["labeled_total"] = labeled_total_;
  model["match"] = match_;
  model["recall"] = recall_;
  model["precision"] = precision_;
  model["timestamp"] = timestamp_;
  json lights;
  for (const auto& it : traffic_light_pairs_) {
    json match;
    match["labeled_light"] = it.first->ToJson();
    match["perception_light"] = it.second->ToJson();
    lights.push_back(match);
  }
  model["match_lights"] = lights;
  return model;
}

TrafficLightDetectionReportModel*
TrafficLightDetectionReportModel::SetTotalPerceptionFrames(int total) {
  total_perception_frames_ = total;
  return this;
}

TrafficLightDetectionReportModel*
TrafficLightDetectionReportModel::SetPrecisionAverage(
    double precision_average) {
  precision_average_ = precision_average;
  return this;
}

TrafficLightDetectionReportModel*
TrafficLightDetectionReportModel::SetRecallAverage(double recall_average) {
  recall_average_ = recall_average;
  return this;
}

TrafficLightDetectionReportModel*
TrafficLightDetectionReportModel::SetTotalLabeledFrames(int total) {
  total_labeled_frames_ = total;
  return this;
}

TrafficLightDetectionReportModel*
TrafficLightDetectionReportModel::SetTotalMatchFrames(int total) {
  total_match_frames_ = total;
  return this;
}

TrafficLightDetectionReportModel* TrafficLightDetectionReportModel::SetWeather(
    std::string weather) {
  weather_ = weather;
  return this;
}

json TrafficLightDetectionReportModel::ToJson() {
  json report;
  report["perception_traffic_lights_total"] = perception_traffic_lights_total_;
  report["labeled_traffic_lights_total"] = labeled_traffic_lights_total_;
  report["match_traffic_lights"] = match_traffic_lights_;
  report["total_perception_frames"] = total_perception_frames_;
  report["precision_average"] = precision_average_;
  report["recall_average"] = recall_average_;
  report["total_labeled_frames"] = total_labeled_frames_;
  report["total_match_frames"] = total_match_frames_;
  // report["weather"] = weather_;
  report["countdown_time_recall"] = countdown_time_recall_;
  report["countdown_time_precision"] = countdown_time_precision_;
  json lights;
  for (const auto& it : matches_) {
    lights[std::to_string(it->GetTimestamp())] = it->ToJson();
  }
  report["match_lights"] = lights;
  return report;
}

bool TrafficLightDetectionReportModel::SerializeToFile(
    const std::string& file) {
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

TrafficLightDetectionReportModel*
TrafficLightDetectionReportModel::Calculate() {
  match_traffic_lights_ = 0;
  perception_traffic_lights_total_ = 0;
  labeled_traffic_lights_total_ = 0;
  total_match_frames_ = matches_.size();
  int countdown_light_match = 0;
  int perception_countdown_lights = 0;
  int labeled_countdown_lights = 0;
  for (const auto& it : matches_) {
    match_traffic_lights_ += it->match_;
    perception_traffic_lights_total_ += it->perception_total_;
    labeled_traffic_lights_total_ += it->labeled_total_;
    countdown_light_match += it->countdown_light_match_;
    perception_countdown_lights += it->perception_countdown_lights_;
    labeled_countdown_lights += it->labeled_countdown_lights_;
  }
  if (perception_traffic_lights_total_ == 0) {
    precision_average_ = 0;
  } else {
    precision_average_ = match_traffic_lights_ /
                         static_cast<double>(perception_traffic_lights_total_);
  }
  if (labeled_traffic_lights_total_ == 0) {
    recall_average_ = 0;
  } else {
    recall_average_ = match_traffic_lights_ /
                      static_cast<double>(labeled_traffic_lights_total_);
  }
  if (perception_countdown_lights > 0) {
    countdown_time_precision_ =
        countdown_light_match /
        static_cast<double>(perception_countdown_lights);
  } else {
    countdown_time_precision_ = 0;
  }
  if (labeled_countdown_lights > 0) {
    countdown_time_recall_ =
        countdown_light_match / static_cast<double>(labeled_countdown_lights);
  } else {
    countdown_time_recall_ = 0;
  }
  AINFO << "countdown_light_match = " << countdown_light_match
        << " perception_countdown_lights = " << perception_countdown_lights
        << " labeled_countdown_lights = " << labeled_countdown_lights;
  return this;
}

TrafficLightDetectionReportModel* TrafficLightDetectionReportModel::ShowInfo() {
  AINFO << "perception_traffic_lights_total = "
        << perception_traffic_lights_total_
        << ". labeled_traffic_lights_total_total = "
        << labeled_traffic_lights_total_
        << " match_traffic_lights = " << match_traffic_lights_
        << ". total_perception_frames = " << total_perception_frames_
        << ". precision_average = " << precision_average_
        << ". recall_average = " << recall_average_
        << ". total_labeled_frames = " << total_labeled_frames_
        << ". total_match_frames = " << total_match_frames_
        << ". countdown_time_recall = " << countdown_time_recall_
        << ". countdown_time_precision = " << countdown_time_precision_;
  return this;
}

TrafficLightDetectionReportModel*
TrafficLightDetectionReportModel::AddTrafficLightFramePtr(
    const TrafficLightsFramePtr& item) {
  matches_.push_back(item);
  return this;
}

}  // namespace integration_test
}  // namespace roadstar
