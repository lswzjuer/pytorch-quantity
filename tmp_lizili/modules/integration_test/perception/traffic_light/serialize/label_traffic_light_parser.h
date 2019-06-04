#ifndef MODULES_INTEGRATION_TEST_PERCEPTION_TRAFFIC_SERIALIZE_LABEL_TRAFFIC_LGIHT_PARSER_H
#define MODULES_INTEGRATION_TEST_PERCEPTION_TRAFFIC_SERIALIZE_LABEL_TRAFFIC_LGIHT_PARSER_H

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "modules/common/util/file.h"
#include "modules/integration_test/perception/traffic_light/model/traffic_light_model.h"
#include "third_party/json/json.hpp"

namespace roadstar {
namespace integration_test {

using json = nlohmann::json;

class LabelTrafficLightParser {
 public:
  LabelTrafficLightParser(const std::string& path, const std::string& save_path)
      : path_(path), save_path_(save_path) {
    ParseJson();
  }

  const LabeledTrafficLightDetectionModelPtrVec& GetTrafficLights() const;
  bool Save();
  bool ParseJson();

 private:
  void ReplaceAndTransform(const json& in_json);
  json TrafficLightTransform(const json& value);
  std::string path_;
  std::string save_path_;
  LabeledTrafficLightDetectionModelPtrVec traffic_lights_;
};

}  // namespace integration_test
}  // namespace roadstar

#endif  // MODULES_INTEGRATION_TEST_PERCEPTION_OBSTACLE_SERIALIZE_LABEL_DATA_TRANSFORMER_H
