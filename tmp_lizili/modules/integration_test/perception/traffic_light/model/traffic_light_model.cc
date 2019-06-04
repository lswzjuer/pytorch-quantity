#include "modules/integration_test/perception/traffic_light/model/traffic_light_model.h"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

#include "modules/common/log.h"
#include "modules/msgs/perception/proto/traffic_light_detection.pb.h"

namespace roadstar {
namespace integration_test {

TrafficLightModel::TrafficLightModel(const TrafficLight& msg) {
  FromTrafficLightMsg(msg);
}

TrafficLightModel::TrafficLightModel(const json& value) {
  FromJson(value);
}
int TrafficLightModel::GetLightType() const {
  return light_type_;
}

int TrafficLightModel::GetColor() const {
  return color_;
}

Eigen::Vector4d TrafficLightModel::GetBox4d() const {
  return box4d_;
}

TrafficLightModel* TrafficLightModel::SetIgnore(int ignore) {
  ignore_ = ignore;
  return this;
}

bool TrafficLightModel::IsIgnore() const {
  return ignore_ == 1;
}

TrafficLightModel* TrafficLightModel::SetColor(int color) {
  color_ = color;
  return this;
}

TrafficLightModel* TrafficLightModel::SetBox4d(const Eigen::Vector4d& box) {
  box4d_ = box;
  return this;
}

TrafficLightModel* TrafficLightModel::SetLightType(int type) {
  light_type_ = type;
  return this;
}

int TrafficLightModel::GetCountDownTime() const {
  return countdown_time_;
}

TrafficLightModel* TrafficLightModel::SetCountDownTime(
    const int& countdown_time) {
  countdown_time_ = countdown_time;
  return this;
}

json TrafficLightModel::ToJson() {
  json out;
  out["color"] = color_;
  out["light_type"] = light_type_;
  out["countdown_time"] = countdown_time_;
  json box;
  box["xmin"] = box4d_[0];
  box["ymin"] = box4d_[1];
  box["xmax"] = box4d_[2];
  box["ymax"] = box4d_[3];
  out["img_box"] = box;
  return out;
}

TrafficLightModel& TrafficLightModel::FromJson(const json& out) {
  try {
    color_ = out["color"];
    light_type_ = out["light_type"];
    if (out.find("countdown_time") != out.end()) {
      countdown_time_ = out["countdown_time"];
    } else {
      countdown_time_ = -1;
    }
    json box = out["img_box"];
    box4d_[0] = box["xmin"];
    box4d_[1] = box["ymin"];
    box4d_[2] = box["xmax"];
    box4d_[3] = box["ymax"];
    if (out.find("ignore") != out.end()) {
      ignore_ = out["ignore"];
    } else {
      ignore_ = 0;
    }
  } catch (...) {
    AERROR << "catch error when try to cast model from json.";
  }
  return *this;
}

void TrafficLightModel::FromTrafficLightMsg(const TrafficLight& msg) {
  color_ = msg.color();
  light_type_ = msg.light_type();
  countdown_time_ = msg.countdown_time();
  const auto& box = msg.img_box();
  box4d_[0] = box.xmin();
  box4d_[1] = box.ymin();
  box4d_[2] = box.xmax();
  box4d_[3] = box.ymax();
  ignore_ = 0;
}

void TrafficLightModel::ToTrafficLightMsg(TrafficLight* msg) const {
  msg->set_color(static_cast<roadstar::perception::TrafficLight_Color>(color_));
  msg->set_light_type(
      static_cast<roadstar::perception::TrafficLight_LightType>(light_type_));
  auto* box = msg->mutable_img_box();
  box->set_xmin(box4d_[0]);
  box->set_ymin(box4d_[1]);
  box->set_xmax(box4d_[2]);
  box->set_ymax(box4d_[3]);
  msg->set_countdown_time(countdown_time_);
}

template <class Flag>
TrafficLightDetectionModel<Flag>&
TrafficLightDetectionModel<Flag>::FromTrafficLightDetectionMsg(
    const TrafficLightDetection& msg) {
  timestamp_ = msg.header().timestamp_sec();
  const auto& roi = msg.roi();
  roi_[0] = roi.xmin();
  roi_[1] = roi.ymin();
  roi_[2] = roi.xmax();
  roi_[3] = roi.ymax();
  uint32_t size = msg.traffic_light_size();
  for (auto i = 0; i < size; ++i) {
    traffic_lights_.emplace_back(msg.traffic_light(i));
  }
  return *this;
}

template <class Flag>
json TrafficLightDetectionModel<Flag>::ToJson() {
  json value;
  value["timestamp"] = timestamp_;
  json roi;
  roi["xmin"] = roi_[0];
  roi["ymin"] = roi_[1];
  roi["xmax"] = roi_[2];
  roi["ymax"] = roi_[3];
  value["roi"] = roi;
  json traffic_light;
  for (auto& it : traffic_lights_) {
    traffic_light.push_back(it.ToJson());
  }
  value["traffic_light"] = traffic_light;
  return value;
}

template <class Flag>
std::size_t TrafficLightDetectionModel<Flag>::TrafficLightSize() const {
  return traffic_lights_.size();
}

template <class Flag>
std::size_t TrafficLightDetectionModel<Flag>::CountTimeTrafficLightSize()
    const {
  std::size_t size = 0;
  for (auto& it : traffic_lights_) {
    if (it.GetCountDownTime() >= 0) {
      ++size;
    }
  }
  return size;
}

template <class Flag>
TrafficLightDetectionModel<Flag>& TrafficLightDetectionModel<Flag>::FromJson(
    const json& value) {
  try {
    std::string timestamp = value["timestamp"];
    timestamp_ = std::stod(timestamp);
    if (value.find("roi") != value.end()) {
      json roi = value["roi"];
      roi_[0] = roi["xmin"];
      roi_[1] = roi["ymin"];
      roi_[2] = roi["xmax"];
      roi_[3] = roi["ymax"];
    } else {
      roi_[0] = 0;
      roi_[1] = 0;
      roi_[2] = 0;
      roi_[3] = 0;
    }
    json traffic_light = value["traffic_light"];
    for (auto& it : traffic_light) {
      traffic_lights_.push_back(TrafficLightModel(it));
    }
  } catch (...) {
    AERROR << "catch error when cast from json.";
    return *this;
  }
  return *this;
}

template <class Flag>
void TrafficLightDetectionModel<Flag>::SetTimestamp(double timestamp) {
  timestamp_ = timestamp;
}

template <class Flag>
void TrafficLightDetectionModel<Flag>::SetRoi(const Eigen::Vector4d& roi) {
  roi_ = roi;
}

template <class Flag>
void TrafficLightDetectionModel<Flag>::AddTrafficLight(
    const TrafficLightModel& model) {
  traffic_lights_.push_back(model);
}

template <class Flag>
double TrafficLightDetectionModel<Flag>::GetTimestamp() const {
  return timestamp_;
}

template <class Flag>
const Eigen::Vector4d& TrafficLightDetectionModel<Flag>::GetRoi() const {
  return roi_;
}

template <class Flag>
const TrafficLightModel& TrafficLightDetectionModel<Flag>::GetTrafficLight(
    uint32_t index) const {
  return traffic_lights_[index];
}

template <class Flag>
TrafficLightModel* TrafficLightDetectionModel<Flag>::MutableTrafficLight(
    uint32_t index) {
  return &traffic_lights_[index];
}

template <class Flag>
bool TrafficLightDetectionModel<Flag>::SerializeToFile(
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

template class TrafficLightDetectionModel<PerceptionDetectionFlag>;
template class TrafficLightDetectionModel<LabeledDetectionFlag>;

}  // namespace integration_test
}  // namespace roadstar
