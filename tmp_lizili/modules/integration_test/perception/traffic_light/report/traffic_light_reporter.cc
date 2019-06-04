#include "modules/integration_test/perception/traffic_light/report/traffic_light_reporter.h"

#include <map>

#include "modules/common/log.h"
#include "modules/common/util/file.h"
#include "modules/integration_test/perception/common/model/config_model.h"
#include "modules/integration_test/perception/data_manager/integration_test_data_manager.h"
#include "modules/integration_test/perception/traffic_light/match/traffic_light_match_impl.h"
#include "modules/integration_test/perception/traffic_light/match/traffic_light_match_interface.h"
#include "modules/integration_test/perception/traffic_light/model/traffic_light_detection_report_model.h"
#include "modules/integration_test/perception/traffic_light/model/traffic_light_model.h"

namespace roadstar {
namespace integration_test {

void TrafficLightReporter::GenerateReport() {
  const auto perception_traffic_lights =
      IntegrationTestDataManager::instance()->GetPerceptionTrafficLightModels();
  const auto labeled_traffic_lights =
      IntegrationTestDataManager::instance()->GetLabeledTrafficLightModels();
  std::unique_ptr<TrafficLightMatchInterface> matcher(
      new TrafficLightMatchImpl(configs_));
  auto report_model =
      matcher->Match(perception_traffic_lights, labeled_traffic_lights);
  std::string reporter_path =
      configs_->GetValueViaKey("traffic_light_report_path");
  roadstar::common::util::EnsureDirectory(reporter_path);
  std::string reporter_name =
      configs_->GetValueViaKey("traffic_light_report_name");
  std::string file = reporter_path + "/" + reporter_name;
  if (report_model) {
    AINFO << "GenerateReport begin "
          << "reporter_path = " << reporter_path
          << " reporter_name = " << reporter_name;
    report_model->SerializeToFile(file);
  }
}

}  // namespace integration_test
}  // namespace roadstar
