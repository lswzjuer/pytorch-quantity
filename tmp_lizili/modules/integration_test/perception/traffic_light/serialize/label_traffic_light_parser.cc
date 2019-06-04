#include "modules/integration_test/perception/traffic_light/serialize/label_traffic_light_parser.h"

#include <sstream>

#include "modules/common/log.h"

namespace roadstar {
namespace integration_test {

void LabelTrafficLightParser::ReplaceAndTransform(const json& in_json) {
  try {
    for (auto it : in_json) {
      json traffic_light = TrafficLightTransform(it["traffic_light"]);
      it["traffic_light"] = traffic_light;
      LabeledTrafficLightDetectionModelPtr ptr(
          new LabeledTrafficLightDetectionModel(it));
      traffic_lights_.push_back(ptr);
    }
  } catch (...) {
    AERROR << "json parse error";
  }
}

const LabeledTrafficLightDetectionModelPtrVec&
LabelTrafficLightParser::GetTrafficLights() const {
  return traffic_lights_;
}

bool LabelTrafficLightParser::ParseJson() {
  try {
    std::ifstream in_file(path_);
    if (!in_file.is_open()) {
      AERROR << "file open error. path is " << path_;
      return false;
    }
    json value;
    in_file >> value;
    in_file.close();
    ReplaceAndTransform(value);
  } catch (...) {
    AERROR << "Catch exception.";
    return false;
  }
  return true;
}

json LabelTrafficLightParser::TrafficLightTransform(const json& value) {
  json out;
  for (auto& it : value) {
    json light;
    int light_class = it["class"];
    light["color"] = light_class / 10;
    light["light_type"] = light_class % 10;
    json img_box;
    img_box["xmin"] = it["x1"];
    img_box["xmax"] = it["x2"];
    img_box["ymin"] = it["y1"];
    img_box["ymax"] = it["y2"];
    light["img_box"] = img_box;
    light["ignore"] = it["ignore"];
    light["countdown_time"] = it["countdown_time"];
    // AINFO << "light['ignore'] = " << light["ignore"];
    out.push_back(light);
  }
  return out;
}

bool LabelTrafficLightParser::Save() {
  if (save_path_.length() == 0) {
    AERROR << "Aerror.Save labeled data fail cause save_path is empty.";
    return false;
  }
  roadstar::common::util::EnsureDirectory(save_path_);
  for (auto& it : traffic_lights_) {
    std::string sub_file(save_path_ + "/" + std::to_string(it->GetTimestamp()) +
                         ".json");
    it->SerializeToFile(sub_file);
  }
  AINFO << "Save json to file successfully and the output path is "
        << save_path_.c_str() << std::endl;
  return true;
}

}  // namespace integration_test
}  // namespace roadstar
