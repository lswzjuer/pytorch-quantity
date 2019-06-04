#ifndef MODULES_INTEGRATION_TEST_PERCEPTION_OBSTACLE_SERIALIZE_LABEL_DATA_PARSER_H
#define MODULES_INTEGRATION_TEST_PERCEPTION_OBSTACLE_SERIALIZE_LABEL_DATA_PARSER_H

#include <string>
#include <map>
#include <fstream>
#include <iostream>
#include <vector>

#include "modules/common/util/file.h"
#include "modules/integration_test/perception/obstacle/model/label_frame_model.h"
#include "modules/integration_test/perception/obstacle/model/labeled_obstacle_model.h"
#include "third_party/json/json.hpp"

namespace roadstar {
namespace integration_test {

using json = nlohmann::json;

class LabelDataParser {
 public:
  LabelDataParser(const std::string& path, const std::string& save_path)
      : path_(path), save_path_(save_path), max_frame_(0) {}

  bool ParseAndSplitByFrame(unsigned int* first_box_id);
  std::vector<LabelFrameModel> ParseFramesData();
  bool Save();
  std::map<int, LabelFrameModel>& GetLabelObstacles();
  std::vector<LabelFrameModel> GetLabelObstaclesVectorStyle();

 private:
  void SplitByFrame(const json& velodyne64_in, unsigned int* first_box_id);
  std::string path_;
  std::string save_path_;
  int max_frame_;
  std::map<int, LabelFrameModel> map_obstacles_;
};

}  // namespace integration_test
}  // namespace roadstar

#endif  // MODULES_INTEGRATION_TEST_PERCEPTION_OBSTACLE_SERIALIZE_LABEL_DATA_TRANSFORMER_H
