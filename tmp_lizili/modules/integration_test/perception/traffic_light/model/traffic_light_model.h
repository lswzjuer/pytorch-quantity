#ifndef MODULES_INTEGRATION_TEST_PERCEPTION_TRAFFIC_LIGHT_MODEL_TRAFFIC_LIGHT_MODEL_H
#define MODULES_INTEGRATION_TEST_PERCEPTION_TRAFFIC_LIGHT_MODEL_TRAFFIC_LIGHT_MODEL_H

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "Eigen/Core"

#include "modules/common/proto/geometry.pb.h"
#include "modules/msgs/perception/proto/traffic_light_detection.pb.h"
#include "third_party/json/json.hpp"

namespace roadstar {
namespace integration_test {

using json = nlohmann::json;
using roadstar::common::Box2d;
using roadstar::perception::TrafficLight;
using roadstar::perception::TrafficLightDetection;

class TrafficLightModel {
 public:
  TrafficLightModel() = default;
  explicit TrafficLightModel(const json& value);
  explicit TrafficLightModel(const TrafficLight& msg);
  int GetLightType() const;
  int GetColor() const;
  int GetCountDownTime() const;
  bool IsIgnore() const;
  Eigen::Vector4d GetBox4d() const;
  TrafficLightModel* SetColor(int color);
  TrafficLightModel* SetIgnore(int ingnore);
  TrafficLightModel* SetLightType(int type);
  TrafficLightModel* SetBox4d(const Eigen::Vector4d& box);
  TrafficLightModel* SetCountDownTime(const int& countdown_time);
  json ToJson();
  TrafficLightModel& FromJson(const json& out);
  void FromTrafficLightMsg(const TrafficLight& msg);
  void ToTrafficLightMsg(TrafficLight* msg) const;

 private:
  int countdown_time_ = -1;
  int light_type_;
  int color_;
  Eigen::Vector4d box4d_;
  int ignore_ = 0;
};

struct PerceptionDetectionFlag {};
struct LabeledDetectionFlag {};

template <class Flag>
class TrafficLightDetectionModel {
 public:
  TrafficLightDetectionModel() = default;
  explicit TrafficLightDetectionModel(const TrafficLightDetection& msg) {
    FromTrafficLightDetectionMsg(msg);
  }
  explicit TrafficLightDetectionModel(const json& value) {
    FromJson(value);
  }
  void SetTimestamp(double timestamp);
  void AddTrafficLight(const TrafficLightModel& model);
  double GetTimestamp() const;
  const TrafficLightModel& GetTrafficLight(uint32_t index) const;
  std::size_t TrafficLightSize() const;
  std::size_t CountTimeTrafficLightSize() const;
  TrafficLightModel* MutableTrafficLight(uint32_t index);
  json ToJson();
  TrafficLightDetectionModel& FromJson(const json& value);
  TrafficLightDetectionModel& FromTrafficLightDetectionMsg(
      const TrafficLightDetection& msg);
  bool SerializeToFile(const std::string& file);
  const Eigen::Vector4d& GetRoi() const;
  void SetRoi(const Eigen::Vector4d& roi);

 private:
  std::vector<TrafficLightModel> traffic_lights_;
  double timestamp_;
  Eigen::Vector4d roi_;
};

typedef TrafficLightDetectionModel<PerceptionDetectionFlag>
    PerceptionTrafficLightDetectionModel;
typedef TrafficLightDetectionModel<LabeledDetectionFlag>
    LabeledTrafficLightDetectionModel;

typedef std::shared_ptr<PerceptionTrafficLightDetectionModel>
    PerceptionTrafficLightDetectionModelPtr;
typedef std::shared_ptr<LabeledTrafficLightDetectionModel>
    LabeledTrafficLightDetectionModelPtr;
typedef std::vector<LabeledTrafficLightDetectionModelPtr>
    LabeledTrafficLightDetectionModelPtrVec;
typedef std::vector<PerceptionTrafficLightDetectionModelPtr>
    PerceptionTrafficLightDetectionModelPtrVec;
}  // namespace integration_test
}  // namespace roadstar

#endif  // MODULES_INTEGRATION_TEST_PERCEPTION_TRAFFIC_LIGHT_MODEL_TRAFFIC_LIGHT_MODEL_H
