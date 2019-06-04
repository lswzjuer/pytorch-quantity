#include "modules/integration_test/perception/statistical_analysis/statistical_analysis_processes.h"

#include <string>

#include "modules/common/log.h"
#include "modules/integration_test/perception/common/test_object.h"
#include "modules/integration_test/perception/statistical_analysis/obstacle_analysis_impl.h"
#include "modules/integration_test/perception/statistical_analysis/traffic_light_analysis_impl.h"

namespace roadstar {
namespace integration_test {

StatisticalAnalysisProcesses::StatisticalAnalysisProcesses(
    const std::shared_ptr<ConfigModel>& configs) {
  std::string test_object = configs->GetValueViaKey("test_object");
  AINFO << "StatisticalAnalysisProcesses begin.test_object = " << test_object;
  int mode = std::stoi(test_object);
  if (TestObject::IsTestObstacleMode(mode)) {
    analyzers_.emplace_back(new ObstacleAnalysisImpl(configs));
  }
  if (TestObject::IsTestTrafficLightMode(mode)) {
    analyzers_.emplace_back(new TrafficLightAnalysisImpl(configs));
  }
}

void StatisticalAnalysisProcesses::StartAnalyze() {
  for (const auto& it : analyzers_) {
    it->Analyze();
  }
}

}  // namespace integration_test
}  // namespace roadstar
