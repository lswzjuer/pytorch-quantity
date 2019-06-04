#ifndef MODULES_INTEGRATION_TEST_PERCEPTION_PERCEPTION_TEST_H_
#define MODULES_INTEGRATION_TEST_PERCEPTION_PERCEPTION_TEST_H_

#include <ros/callback_queue.h>
#include <ros/include/ros/ros.h>

#include <memory>
#include <string>
#include <vector>

#include "modules/common/roadstar_app.h"
#include "modules/integration_test/perception/common/model/config_model.h"
#include "modules/integration_test/perception/callback_manager/subnode_callback_manager.h"

namespace roadstar {
namespace integration_test {

namespace rc = roadstar::common;

class PerceptionTest : public roadstar::common::RoadstarApp {
 public:
  PerceptionTest() = default;

  std::string Name() const override {
    return "perception_test";
  }

  Status Init() override;

  Status Start() override;

  void Stop() override;

  static bool end_successfully_;

 private:
  bool GetConfigs();

 private:
  std::unique_ptr<SubnodeCallbackManager> subnode_callback_manager_;
  std::shared_ptr<ConfigModel> configs_;
};

}  // namespace integration_test
}  // namespace roadstar

#endif  // MODULES_INTEGRATION_TEST_PERCEPTION_PERCEPTION_TEST_H_
