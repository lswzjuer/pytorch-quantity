#ifndef MODULES_PERCEPTION_TEST_PERCEPTION_TRAFFIC_LIGHT_MATCH_TRAFFIC_LIGHT_MATCH_INTERFACE_H
#define MODULES_PERCEPTION_TEST_PERCEPTION_TRAFFIC_LIGHT_MATCH_TRAFFIC_LIGHT_MATCH_INTERFACE_H

#include <map>
#include <memory>

#include "modules/integration_test/perception/common/model/config_model.h"
#include "modules/integration_test/perception/traffic_light/model/traffic_light_detection_report_model.h"
#include "modules/integration_test/perception/traffic_light/model/traffic_light_model.h"

namespace roadstar {
namespace integration_test {

class TrafficLightMatchInterface {
 public:
  virtual TrafficLightDetectionReportModelPtr Match(
      const PerceptionTrafficLightDetectionModelPtrVec
          &perception_traffic_lights_data,
      const LabeledTrafficLightDetectionModelPtrVec
          &labeled_traffic_lights_data) = 0;
  virtual ~TrafficLightMatchInterface() = default;
};

}  // namespace integration_test
}  // namespace roadstar

#endif
