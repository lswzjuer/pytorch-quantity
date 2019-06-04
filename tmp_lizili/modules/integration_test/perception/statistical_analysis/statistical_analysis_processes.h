#ifndef MODULES_PERCEPTION_TEST_PERCEPTION_STATISTICAL_ANALYSIS_STATISTICAL_ANALYSIS_PROCESSES_H
#define MODULES_PERCEPTION_TEST_PERCEPTION_STATISTICAL_ANALYSIS_STATISTICAL_ANALYSIS_PROCESSES_H

#include <memory>
#include <vector>

#include "modules/integration_test/perception/common/model/config_model.h"
#include "modules/integration_test/perception/statistical_analysis/statistical_analysis_interface.h"

namespace roadstar {
namespace integration_test {

class StatisticalAnalysisProcesses {
 public:
  typedef std::unique_ptr<StatisticalAnalysisInterface>
      StatisticalAnalysisInterfacePtr;
  explicit StatisticalAnalysisProcesses(
      const std::shared_ptr<ConfigModel>& configs);
  void StartAnalyze();

 private:
  std::vector<StatisticalAnalysisInterfacePtr> analyzers_;
};

}  // namespace integration_test
}  // namespace roadstar

#endif
