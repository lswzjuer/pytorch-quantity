#include "modules/integration_test/perception/statistical_analysis/traffic_light_analysis_impl.h"

#include <string>

#include "modules/common/log.h"
#include "modules/integration_test/perception/data_manager/integration_test_data_manager.h"
#include "modules/integration_test/perception/traffic_light/report/traffic_light_reporter.h"

namespace roadstar {
namespace integration_test {

TrafficLightAnalysisImpl::TrafficLightAnalysisImpl(
    const std::shared_ptr<ConfigModel>& configs)
    : configs_(configs) {}

TrafficLightAnalysisImpl::~TrafficLightAnalysisImpl() {
  if (analyzer_thread_.joinable()) {
    AINFO << "TrafficLightAnalysisImpl destruct,thread running.";
    analyzer_thread_.join();
    AINFO << "TrafficLightAnalysisImpl destruct,thread completed.";
  }
}
void TrafficLightAnalysisImpl::Analyze() {
  std::thread thread(
      std::bind(&TrafficLightAnalysisImpl::InternalAnalyze, this));
  analyzer_thread_.swap(thread);
}

void TrafficLightAnalysisImpl::InternalAnalyze() {
  AINFO << " begin serialization data ...";
  std::string mid_file_path = configs_->GetValueViaKey("mid_file_path");
  std::string traffic_data_path =
      mid_file_path + "perception_traffic_light_data";
  IntegrationTestDataManager::instance()->SerializeTrafficLightMsgs(
      traffic_data_path);
  TrafficLightReporter reporter(configs_);
  reporter.GenerateReport();
}

}  // namespace integration_test
}  // namespace roadstar
