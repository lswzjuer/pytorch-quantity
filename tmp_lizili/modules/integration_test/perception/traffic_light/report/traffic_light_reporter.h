#ifndef MODULES_INTEGRATION_TEST_PERCEPTION_OBSTACLE_REPORT_TRAFFIC_LIGHT_REPORTER_H
#define MODULES_INTEGRATION_TEST_PERCEPTION_OBSTACLE_REPORT_TRAFFIC_LIGHT_REPORTER_H

#include <memory>
#include <string>
#include <vector>

#include "modules/integration_test/perception/common/model/config_model.h"

namespace roadstar {
namespace integration_test {

class TrafficLightReporter {
 public:
  explicit TrafficLightReporter(const std::shared_ptr<ConfigModel> configs)
      : configs_(configs) {}

  void GenerateReport();

 private:
  std::shared_ptr<ConfigModel> configs_;
};

}  // namespace integration_test
}  // namespace roadstar

#endif  // MODULES_INTEGRATION_TEST_PERCEPTION_OBSTACLE_REPORT_TRAFFIC_LIGHT_REPORTER_H
