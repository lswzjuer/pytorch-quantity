#include "modules/integration_test/perception/statistical_analysis/obstacle_analysis_impl.h"

#include "modules/common/log.h"
#include "modules/common/util/file.h"
#include "modules/integration_test/perception/data_manager/integration_test_data_manager.h"
#include "modules/integration_test/perception/obstacle/report/obstacle_reporter.h"

namespace roadstar {
namespace integration_test {

ObstacleAnalysisImpl::~ObstacleAnalysisImpl() {
  if (analyzer_thread_.joinable()) {
    AINFO << "ObstacleAnalysisImpl destruct,thread running.";
    analyzer_thread_.join();
    AINFO << "ObstacleAnalysisImpl destruct,thread completed.";
  }
}

void ObstacleAnalysisImpl::Analyze() {
  std::thread thread(std::bind(&ObstacleAnalysisImpl::InternalAnalyze, this));
  analyzer_thread_.swap(thread);
}

void ObstacleAnalysisImpl::InternalAnalyze() {
  AINFO << " begin serialization data ...";
  const std::vector<double>& time_stamps = GetFrameTimeStamps();
  std::string save_path = RedressAndSave(time_stamps);
  std::vector<std::string> save_paths({save_path});
  std::string perception_data_path;
  DumpPerceptionData(&perception_data_path);
  GenerateReport(perception_data_path, save_paths);
}

void ObstacleAnalysisImpl::DumpPerceptionData(
    std::string* perception_data_path) {
  AINFO << "DumpPerceptionData begin..." << std::endl
        << "Jenkins INFO: fusion_map_listener exit, save data to mid_files "
           "path now.";
  *perception_data_path =
      IntegrationTestDataManager::instance()->SerializePerceptionResults();
}

std::string ObstacleAnalysisImpl::RedressAndSave(
    const std::vector<double>& time_stamps) {
  std::string save_path =
      configs_->GetValueViaKey("mid_file_path") + "label_obstacle_data";
  roadstar::common::util::EnsureDirectory(save_path);
  std::string perception_version = configs_->GetValueViaKey("perception");
  const auto& label_obstacles_models =
      IntegrationTestDataManager::instance()->GetLabeledObstaclesModel();
  AINFO << "RedressAndSave begin save_path = " << save_path
        << " models size = " << label_obstacles_models.size()
        << " velodyne packets size = " << time_stamps.size()
        << " perception version = " << perception_version;
  std::vector<LabelFrameModel> datas(label_obstacles_models);
  std::size_t frame = 0;
  std::size_t total_obstacles = 0;
  int total_left = 0;
  for (auto& it : datas) {
    bool is_match = false;
    for (; frame < time_stamps.size(); ++frame) {
      double time_stamp = it.GetTimeStamp();
      double velodyne_time_stamp = time_stamps[frame];
      if (time_stamp - 0.001 < velodyne_time_stamp &&
          time_stamp + 0.001 > velodyne_time_stamp) {
        std::size_t size = it.Size();
        total_obstacles += size;
        for (std::size_t index = 0; index < size; ++index) {
          LabeledObstacleModel* obstacle = it.GetVelodyneTypeModelAt(index);
          LabeledObstacleModel ego_front_model =
              obstacle->ToEgoFront(perception_version);
          it.AddObstacleOfEgoFrontTypeModel(ego_front_model);
        }
        std::string file(save_path + "//" + std::to_string(frame) + ".json");
        it.SerializeToFile(file);
        ++total_left;
        frame++;
        is_match = true;
        break;
      }
    }
    if (!is_match) {
      frame = 0;  // search from begin again.
    }
  }
  AINFO << "RedressAndSave  end.  total frame left  = " << total_left
        << " total obstacles = " << total_obstacles;
  return save_path;
}

void ObstacleAnalysisImpl::GenerateReport(
    const std::string& perception_data_path,
    const std::vector<std::string>& save_paths) {
  AINFO << "Jenkins INFO: before reporter... perception_data_path = "
        << perception_data_path;
  ObstacleReporter reporter(save_paths, perception_data_path, configs_);
  reporter.GenerateReport();
}

const std::vector<double>& ObstacleAnalysisImpl::GetFrameTimeStamps() {
  return IntegrationTestDataManager::instance()->GetFrameTimeStamps();
}

}  // namespace integration_test
}  // namespace roadstar
