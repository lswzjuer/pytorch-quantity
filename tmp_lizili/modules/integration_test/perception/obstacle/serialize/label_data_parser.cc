#include "modules/integration_test/perception/obstacle/serialize/label_data_parser.h"
#include "modules/common/log.h"

namespace roadstar {
namespace integration_test {

void LabelDataParser::SplitByFrame(const json& velodyne64_in,
                                   unsigned int* first_box_id) {
  if (!first_box_id) {
    AFATAL << "first_box_id is nullptr";
  }
  try {
    const auto it = velodyne64_in.find("0");
    if (it == velodyne64_in.end()) {
      AERROR << "json parse error cause there isn't key named \"0\"";
      return;
    }
    json zero_obj = velodyne64_in["0"];
    json json_obstacles = zero_obj["obstacles"];
    double velocity = 0;
    double heading = 0;
    for (auto& it : json_obstacles) {
      json boxes = it["boxes"];
      int i_start = it["start"];
      std::string str_type = it["type"];
      int type = atoi(str_type.c_str());
      for (auto& box : boxes) {
        if (box.is_null()) {
          continue;
        }
        const auto Obstacle_vec_it = map_obstacles_.find(i_start);
        if (Obstacle_vec_it == map_obstacles_.end()) {
          LabelFrameModel frame_model;
          LabeledObstacleModel obstacale_model(
              box["x"], box["y"], box["b"], it["l"], it["h"], it["w"], box["r"],
              *first_box_id, velocity, heading, type);
          frame_model.AddObstacleOfVelodyneTypeModel(obstacale_model);
          map_obstacles_[i_start] = frame_model;
        } else {
          LabeledObstacleModel obstacale_model(
              box["x"], box["y"], box["b"], it["l"], it["h"], it["w"], box["r"],
              *first_box_id, velocity, heading, type);
          map_obstacles_[i_start].AddObstacleOfVelodyneTypeModel(
              obstacale_model);
        }
        max_frame_ = i_start > max_frame_ ? i_start : max_frame_;
        ++i_start;
      }
      (*first_box_id)++;
    }
  } catch (...) {
    AERROR << "json parse error";
  }
}

std::map<int, LabelFrameModel>& LabelDataParser::GetLabelObstacles() {
  return map_obstacles_;
}

std::vector<LabelFrameModel> LabelDataParser::GetLabelObstaclesVectorStyle() {
  std::vector<LabelFrameModel> vec_;
  for (int i = 0; i <= max_frame_; ++i) {
    const auto& it = map_obstacles_.find(i);
    if (it != map_obstacles_.end()) {
      vec_.push_back(it->second);
      // AINFO << "push_back in vector,index = " << i;
    }
  }
  return vec_;
}

bool LabelDataParser::ParseAndSplitByFrame(unsigned int* first_box_id) {
  if (!first_box_id) {
    AFATAL << "first_box_id is nullptr";
  }
  try {
    std::ifstream in_file(path_);
    if (!in_file.is_open()) {
      AERROR << "file open error. path is " << path_;
      return false;
    }
    json value;
    in_file >> value;
    SplitByFrame(value, first_box_id);
    in_file.close();
  } catch (...) {
    AERROR << "Catch exception.";
    return false;
  }
  return true;
}

std::vector<LabelFrameModel> LabelDataParser::ParseFramesData() {
  std::vector<LabelFrameModel> models;
  try {
    std::vector<std::string> files = roadstar::common::util::ListFiles(path_);
    for (auto sub_it : files) {
      if (sub_it == "." || sub_it == "..") {
        continue;
      }
      std::string frame_file = path_ + "/" + sub_it;
      LabelFrameModel model;
      model.ParseRawDataFromFile(frame_file);
      models.push_back(model);
    }
  } catch (...) {
    AERROR << "Catch exception.";
    return models;
  }
  return models;
}

bool LabelDataParser::Save() {
  if (save_path_.length() == 0) {
    AERROR << "Aerror.Save labeled data fail cause save_path is empty.";
    return false;
  }
  roadstar::common::util::EnsureDirectory(save_path_);
  for (auto it = map_obstacles_.begin(); it != map_obstacles_.end(); ++it) {
    std::string sub_file(save_path_ + "/" + std::to_string(it->first) +
                         ".json");
    it->second.SerializeRawDataToFile(sub_file);
  }
  AINFO << "Save json to file successfully and the output path is "
        << save_path_.c_str() << std::endl;
  return true;
}

}  // namespace integration_test
}  // namespace roadstar
