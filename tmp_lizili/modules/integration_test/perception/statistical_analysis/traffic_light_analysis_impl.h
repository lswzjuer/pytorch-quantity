#ifndef MODULES_PERCEPTION_TEST_PERCEPTION_STATISTICAL_ANALYSIS_TRAFFIC_LIGHT_ANALYSIS_IMPL_H
#define MODULES_PERCEPTION_TEST_PERCEPTION_STATISTICAL_ANALYSIS_TRAFFIC_LIGHT_ANALYSIS_IMPL_H

#include <memory>
#include <thread>

#include "modules/integration_test/perception/common/model/config_model.h"
#include "modules/integration_test/perception/statistical_analysis/statistical_analysis_interface.h"

namespace roadstar {
namespace integration_test {

class TrafficLightAnalysisImpl : public StatisticalAnalysisInterface {
 public:
  explicit TrafficLightAnalysisImpl(
      const std::shared_ptr<ConfigModel>& configs);
  ~TrafficLightAnalysisImpl();
  explicit TrafficLightAnalysisImpl(const TrafficLightAnalysisImpl&) = delete;
  TrafficLightAnalysisImpl& operator=(const TrafficLightAnalysisImpl&) = delete;
  void Analyze() override;

 private:
  void InternalAnalyze();

 private:
  std::thread analyzer_thread_;
  std::shared_ptr<ConfigModel> configs_;
};

}  // namespace integration_test
}  // namespace roadstar

#endif
