#ifndef MODULES_INTEGRATION_TEST_PERCEPTION_CALLBACK_MANAGER_SUBNODE_CALLBACK_MANAGER_H_
#define MODULES_INTEGRATION_TEST_PERCEPTION_CALLBACK_MANAGER_SUBNODE_CALLBACK_MANAGER_H_

#include <memory>
#include <string>
#include <vector>

#include "modules/common/adapters/adapter_manager.h"
#include "modules/common/log.h"

#include "modules/common/status/status.h"
#include "modules/msgs/localization/proto/localization.pb.h"
#include "modules/msgs/perception/proto/detection_result.pb.h"

#include "modules/integration_test/common/integration_test_gflags.h"
#include "modules/integration_test/perception/common/model/config_model.h"
#include "modules/integration_test/perception/data_manager/integration_test_data_manager.h"
#include "modules/integration_test/perception/proto/integration_test.pb.h"

namespace roadstar {
namespace integration_test {

using Localization = roadstar::localization::Localization;
using Status = roadstar::common::Status;
using AdapterManager = roadstar::common::adapter::AdapterManager;

class SubnodeCallbackManager {
 public:
  SubnodeCallbackManager() = default;
  explicit SubnodeCallbackManager(const std::shared_ptr<ConfigModel> &configs)
      : configs_(configs) {}

  Status Start() {
    IntegrationTestParam params;
    AINFO << "Jenkins INFO:SubnodeCallbackManager start. proto config file = "
          << FLAGS_integration_test_config_file;
    if (!roadstar::common::util::GetProtoFromFile(
            FLAGS_integration_test_config_file, &params)) {
      AERROR << "Jenkins INFO: failed to load perception config file "
             << FLAGS_integration_test_config_file;
      return Status(roadstar::common::ErrorCode::PERCEPTION_ERROR,
                    " Jenkins INFO: failed to load perception config file: " +
                        FLAGS_integration_test_config_file);
    }
    IntegrationTestDataManager::instance()->Init(params, configs_);
    AdapterManager::AddLocalizationCallback(
        &SubnodeCallbackManager::LocalizationCallback, this);
    const auto &subnode_config = params.subnode_config();
    for (const auto &subnode : subnode_config.subnodes()) {
      if (subnode.type() == IntegrationTestParam::DEEP_LIDAR_POST_PROCESS ||
          subnode.type() == IntegrationTestParam::DEEP_LIDAR_SESSION_RUN) {
        IntegrationTestDataManager::instance()->SetDeepLidarEvalMode();
        AdapterManager::AddLidarPerceptionCallback(
            &SubnodeCallbackManager::LidarPerceptionCallback, this);
        AINFO << "Jenkins INFO: IN EVAL MODE";
        break;
      }
    }

    // Load subnode
    for (const auto &subnode : subnode_config.subnodes()) {
      switch (subnode.type()) {
        case IntegrationTestParam::VELO64:
          AdapterManager::AddVelo64PacketsCallback(
              &SubnodeCallbackManager::MetaPointCloudPacketsCallback, this);
          break;
        case IntegrationTestParam::HESAI:
          AdapterManager::AddHESAIPacketsCallback(
              &SubnodeCallbackManager::MetaHESAIPointCloudPacketsCallback,
              this);
          break;
        case IntegrationTestParam::VLP_0:
          AdapterManager::AddVlp1PacketsCallback(
              &SubnodeCallbackManager::MetaPointCloudPacketsCallback, this);
          break;
        case IntegrationTestParam::VLP_1:
          AdapterManager::AddVlp2PacketsCallback(
              &SubnodeCallbackManager::MetaPointCloudPacketsCallback, this);
          break;
        case IntegrationTestParam::VLP_2:
          AdapterManager::AddVlp3PacketsCallback(
              &SubnodeCallbackManager::MetaPointCloudPacketsCallback, this);
          break;
        case IntegrationTestParam::FUSION_MAP:
          AdapterManager::AddFusionMapCallback(
              &SubnodeCallbackManager::FusionMapCallback, this);
          break;
        case IntegrationTestParam::TRAFFIC_LIGHT_DETECTION:
          AdapterManager::AddTrafficLightDetectionCallback(
              &SubnodeCallbackManager::TrafficLightDetectionCallback, this);
          break;
        case IntegrationTestParam::DEEP_LIDAR_SESSION_RUN:
          break;
        case IntegrationTestParam::DEEP_LIDAR_POST_PROCESS:
          break;
        default:
          return Status(roadstar::common::ErrorCode::PERCEPTION_ERROR,
                        "Jenkins INFO: Wrong Subnode type");
          break;
      }
    }
    return Status::OK();
  }

 private:
  void FusionMapCallback(const roadstar::perception::FusionMap &msg) {
    IntegrationTestDataManager::instance()->AddFusionMapMsg(msg);
  }

  void TrafficLightDetectionCallback(
      const roadstar::perception::TrafficLightDetection &msg) {
    IntegrationTestDataManager::instance()->AddTrafficLightDetectionMsg(msg);
  }

  void LidarPerceptionCallback(
      const roadstar::perception::DetectionResultProto &msg) {
    IntegrationTestDataManager::instance()->AddLidarDetectionResult(msg);
  }

  void LocalizationCallback(const Localization &msg) {
    IntegrationTestDataManager::instance()->AddLocalizationMsg(msg);
  }

  void MetaPointCloudPacketsCallback(
      const velodyne_msgs::VelodyneScanUnified &msg) {
    double time = static_cast<double>(msg.header.stamp.sec) +
                  static_cast<double>(msg.header.stamp.nsec) / 1000000000;
    IntegrationTestDataManager::instance()->AddVelodynePacketsTimestamp(time);
  }

  void MetaHESAIPointCloudPacketsCallback(const pandar_msgs::PandarScan &msg) {
    double time = static_cast<double>(msg.header.stamp.sec) +
                  static_cast<double>(msg.header.stamp.nsec) / 1000000000;
    IntegrationTestDataManager::instance()->AddVelodynePacketsTimestamp(time);
  }

 private:
  std::shared_ptr<ConfigModel> configs_;
};

}  // namespace integration_test
}  // namespace roadstar
#endif  // MODULES_INTEGRATION_TEST_PERCEPTION_CALLBACK_MANAGER_SUBNODE_CALLBACK_MANAGER_H_
