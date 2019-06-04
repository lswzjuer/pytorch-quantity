#ifndef MODULES_INTEGRATION_TEST_PERCEPTION_DATA_MANAGER_INTEGRATION_TEST_DATA_MANAGER_H
#define MODULES_INTEGRATION_TEST_PERCEPTION_DATA_MANAGER_INTEGRATION_TEST_DATA_MANAGER_H

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "modules/integration_test/perception/common/model/config_model.h"
#include "modules/integration_test/perception/obstacle/model/label_frame_model.h"
#include "modules/integration_test/perception/obstacle/model/location_model.h"
#include "modules/integration_test/perception/obstacle/model/perception_frame_model.h"
#include "modules/integration_test/perception/obstacle/model/perception_obstacle_model.h"
#include "modules/integration_test/perception/proto/integration_test.pb.h"
#include "modules/integration_test/perception/proto/perception_test_data_set.pb.h"
#include "modules/integration_test/perception/traffic_light/model/traffic_light_model.h"

#include "modules/common/macro.h"
#include "modules/msgs/localization/proto/localization.pb.h"
#include "modules/msgs/perception/proto/detection_result.pb.h"
#include "modules/msgs/perception/proto/fusion_map.pb.h"
#include "modules/msgs/perception/proto/obstacle.pb.h"

#include "third_party/json/json.hpp"
#include "velodyne_msgs/PointCloud.h"
#include "velodyne_msgs/VelodyneScanUnified.h"

namespace roadstar {
namespace integration_test {

using FusionMap = roadstar::perception::FusionMap;
using TrafficLightDetection = roadstar::perception::TrafficLightDetection;
using DetectionResultProto = roadstar::perception::DetectionResultProto;
using Localization = roadstar::localization::Localization;
using Json = nlohmann::json;
using Obstacle = roadstar::perception::Obstacle;

class IntegrationTestDataManager {
 public:
  // IntegrationTestDataManager(const IntegrationTestParam& params,
  // const std::shared_ptr<ConfigModel>& configs)
  // : is_eval_(false), params_(params), configs_(configs) {}
  void Init(const IntegrationTestParam& params,
            const std::shared_ptr<ConfigModel>& configs);
  void AddFusionMapMsg(const FusionMap& msg);
  void AddLidarDetectionResult(const DetectionResultProto& msg);
  void AddVelodynePacketsTimestamp(double time_stamp);
  void AddLocalizationMsg(const localization::Localization local);
  void AddTrafficLightDetectionMsg(const TrafficLightDetection& msg);

  void SerializePerceptionMsgs(const std::string& path);
  void SerializeFusionMapMsgs(const std::string& path);
  void SerializeTrafficLightMsgs(const std::string& path);
  void SerializeDeepLidarEvalMsgs(const std::string& path);
  void SerializeLocalization(const std::string& path);
  void SetDeepLidarEvalMode() {
    is_eval_ = true;
  }

  const std::vector<double>& GetFrameTimeStamps();

  const std::vector<LabelFrameModel>& GetLabeledObstaclesModel() const;
  std::string SerializePerceptionResults();
  bool PrepareData();
  const LabeledTrafficLightDetectionModelPtrVec GetLabeledTrafficLightModels()
      const;
  const PerceptionTrafficLightDetectionModelPtrVec GetPerceptionTrafficLightModels()
      const;

 private:
  std::size_t GetFrame(const FusionMap& msg, const std::size_t& begin);
  Localization* GetNearestLocalization(const FusionMap& msg,
                                       const std::size_t& begin);
  void PackageFusionMap();
  bool DumpLabelObstacleData();
  bool DumpLabelTrafficLightData();

 private:
  static bool is_inited_;
  PerceptionTestDataSet perception_test_data_set_;
  std::map<int, PerceptionFrameModel> obstacle_frames_;
  std::vector<double> point_clouds_timestamps_;
  std::vector<double> velodyne_packets_timestamps_;
  std::vector<LabelFrameModel> label_obstacles_models_;
  LabeledTrafficLightDetectionModelPtrVec labeled_traffic_lights_models_;
  std::string mid_files_path_;
  bool is_eval_ = false;
  int perception_obs_ = 0;
  IntegrationTestParam params_;
  std::shared_ptr<ConfigModel> configs_;

  DECLARE_SINGLETON(IntegrationTestDataManager);
};

}  // namespace integration_test
}  // namespace roadstar

#endif  // MODULES_INTEGRATION_TEST_PERCEPTION_DATA_MANAGER_INTEGRATION_TEST_DATA_MANAGER_H
