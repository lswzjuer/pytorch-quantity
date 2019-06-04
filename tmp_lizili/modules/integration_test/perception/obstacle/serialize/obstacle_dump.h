#ifndef MODULES_INTEGRATION_TEST_PERCEPTION_OBSTACLE_SERIALIZE_OBSTACLE_DUMP_H
#define MODULES_INTEGRATION_TEST_PERCEPTION_OBSTACLE_SERIALIZE_OBSTACLE_DUMP_H

#include <string>
#include <vector>
#include <map>

#include "modules/integration_test/perception/obstacle/model/label_frame_model.h"
#include "modules/integration_test/perception/obstacle/model/labeled_obstacle_model.h"
#include "modules/integration_test/perception/obstacle/model/perception_frame_model.h"
#include "modules/integration_test/perception/obstacle/model/perception_obstacle_model.h"

namespace roadstar {
namespace integration_test {

class ObstacleDump {
 public:
  ObstacleDump(const std::vector<std::string>& label_json_pathes,
               const std::string& perception_json_path)
      : label_json_pathes_(label_json_pathes),
        perception_json_path_(perception_json_path) {}

  void Dump();
  std::map<int, LabelFrameModel>& GetLabelObstacles();
  std::map<int, PerceptionFrameModel>& GetPerceptionObstacles();

 private:
  void DumpLabelData();
  void DumpPerceptionData();

 private:
  std::vector<std::string> label_json_pathes_;
  std::string perception_json_path_;

  std::map<int, LabelFrameModel> labeled_obstacles_;
  std::map<int, PerceptionFrameModel> perception_obstacles_;
};
}  // namespace integration_test
}  // namespace roadstar

#endif  // MODULES_INTEGRATION_TEST_PERCEPTION_OBSTACLE_DUMP_H
