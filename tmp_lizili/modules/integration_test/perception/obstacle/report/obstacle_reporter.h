#ifndef MODULES_INTEGRATION_TEST_PERCEPTION_OBSTACLE_REPORT_OBSTACLE_REPORTER_H
#define MODULES_INTEGRATION_TEST_PERCEPTION_OBSTACLE_REPORT_OBSTACLE_REPORTER_H

#include <memory>
#include <vector>
#include <string>

#include "modules/integration_test/perception/common/model/config_model.h"
#include "modules/integration_test/perception/obstacle/model/label_frame_model.h"
#include "modules/integration_test/perception/obstacle/model/labeled_obstacle_model.h"
#include "modules/integration_test/perception/obstacle/model/perception_frame_model.h"
#include "modules/integration_test/perception/obstacle/model/perception_obstacle_model.h"

namespace roadstar {
namespace integration_test {

class ObstacleReporter {
 public:
  ObstacleReporter(const std::vector<std::string>& label_json_pathes,
                   const std::string& perception_json_path,
                   const std::shared_ptr<ConfigModel> configs)
      : label_json_pathes_(label_json_pathes),
        perception_json_path_(perception_json_path),
        configs_(configs) {}

  void GenerateReport();

 private:
  std::vector<std::string> label_json_pathes_;
  std::string perception_json_path_;
  std::shared_ptr<ConfigModel> configs_;
};

}  // namespace integration_test
}  // namespace roadstar

#endif  // MODULES_INTEGRATION_TEST_PERCEPTION_OBSTACLE_REPORTER_H
