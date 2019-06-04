#ifndef MODULES_PERCEPTION_TEST_PERCEPTION_STATISTICAL_ANALYSIS_OBSTACLE_ANALYSIS_IMPL_H
#define MODULES_PERCEPTION_TEST_PERCEPTION_STATISTICAL_ANALYSIS_OBSTACLE_ANALYSIS_IMPL_H

#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "modules/integration_test/perception/common/model/config_model.h"
#include "modules/integration_test/perception/statistical_analysis/statistical_analysis_interface.h"

namespace roadstar {
namespace integration_test {

class ObstacleAnalysisImpl : public StatisticalAnalysisInterface {
 public:
  explicit ObstacleAnalysisImpl(const std::shared_ptr<ConfigModel>& configs)
      : configs_(configs) {}
  ~ObstacleAnalysisImpl();
  ObstacleAnalysisImpl(const ObstacleAnalysisImpl&) = delete;
  ObstacleAnalysisImpl& operator=(const ObstacleAnalysisImpl&) = delete;

  void Analyze() override;

 private:
  void InternalAnalyze();

  void DumpPerceptionData(std::string* perception_data_path);
  std::string RedressAndSave(const std::vector<double>& time_stamps);

  void GenerateReport(const std::string& perception_data_path,
                      const std::vector<std::string>& save_paths);

  const std::vector<double>& GetFrameTimeStamps();

 private:
  std::thread analyzer_thread_;
  std::shared_ptr<ConfigModel> configs_;
};

}  // namespace integration_test
}  // namespace roadstar

#endif
