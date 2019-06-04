#include "modules/integration_test/perception/obstacle/serialize/obstacle_dump.h"
#include "modules/common/log.h"
#include "modules/common/util/file.h"
#include "modules/common/util/string_tokenizer.h"

namespace roadstar {
namespace integration_test {

void ObstacleDump::DumpLabelData() {
  for (auto it : label_json_pathes_) {
    AINFO << "label path is " << it.c_str() << std::endl;
    std::vector<std::string> files = roadstar::common::util::ListFiles(it);
    int size = labeled_obstacles_.size();
    for (const auto& sub_it : files) {
      if (sub_it == "." || sub_it == "..") {
        continue;
      }
      std::string frame_file = it + "/" + sub_it;
      LabelFrameModel model;
      model.ParseFromFile(frame_file);
      std::vector<std::string> json_names =
          roadstar::common::util::StringTokenizer::Split(sub_it, ".");
      if (json_names.size() == 0) {
        continue;
      }
      int key = size + atoi(json_names[0].c_str());
      labeled_obstacles_[key] = model;
    }
  }
  AINFO << "DumpLabelData completed.";
}

void ObstacleDump::DumpPerceptionData() {
  try {
    std::vector<std::string> files =
        roadstar::common::util::ListFiles(perception_json_path_);
    AINFO << "DumpPerceptionData begin.Path is " << perception_json_path_;
    for (const auto& sub_it : files) {
      if (sub_it == "." || sub_it == "..") {
        continue;
      }
      std::string frame_file = perception_json_path_ + "/" + sub_it;
      PerceptionFrameModel frame_model;
      frame_model.ParseFromFile(frame_file);
      std::vector<std::string> json_names =
          roadstar::common::util::StringTokenizer::Split(sub_it, ".");
      if (json_names.size() == 0) {
        continue;
      }

      int key = atoi(json_names[0].c_str());
      perception_obstacles_[key] = frame_model;
    }
    AINFO << "DumpPerceptionData completed. perception_obstacles_ size = "
          << perception_obstacles_.size();
  } catch (...) {
    AERROR << "Error.DumpPerceptionData fail.";
  }
}

void ObstacleDump::Dump() {
  DumpLabelData();
  DumpPerceptionData();
}

std::map<int, LabelFrameModel>& ObstacleDump::GetLabelObstacles() {
  return labeled_obstacles_;
}
std::map<int, PerceptionFrameModel>& ObstacleDump::GetPerceptionObstacles() {
  return perception_obstacles_;
}

}  // namespace integration_test
}  // namespace roadstar
