#include "modules/integration_test/perception/perception_test.h"
#include <stdlib.h>

#include "ros/include/ros/ros.h"

#include "modules/common/adapters/adapter_manager.h"
#include "modules/common/log.h"
#include "modules/common/proto/error_code.pb.h"
#include "modules/integration_test/common/xml_param/xml_param_reader.h"
#include "modules/integration_test/perception/statistical_analysis/statistical_analysis_processes.h"

namespace roadstar {
namespace integration_test {

namespace {

#define SIG_MY_DEFINE_TEST (__SIGRTMIN + 10)
void CustomSigHandler(int signal_num) {
  AINFO << "Jenkins INFO: Received signal: " << signal_num;
  static bool is_stopping = false;
  if (is_stopping) {
    return;
  }
  if (signal_num == SIG_MY_DEFINE_TEST) {
    is_stopping = true;
    PerceptionTest::end_successfully_ = true;
    ros::shutdown();
    return;
  }
}

}  // namespace

bool PerceptionTest::end_successfully_ = false;

Status PerceptionTest::Init() {
  AINFO << "Jenkins INFO: loading xml";
  if (!GetConfigs()) {
    return rc::Status(rc::INTEGRATION_TEST_CONFIG_XML_ERROR,
                      "read config xml failed!");
  }
  signal(SIG_MY_DEFINE_TEST, CustomSigHandler);
  subnode_callback_manager_.reset(new SubnodeCallbackManager(configs_));
  AINFO << "integration_test_adapter_config_file = "
        << FLAGS_integration_test_adapter_config_file;
  rc::adapter::AdapterManager::Init(FLAGS_integration_test_adapter_config_file);
  return subnode_callback_manager_->Start();
}

Status PerceptionTest::Start() {
  bool success = IntegrationTestDataManager::instance()->PrepareData();
  if (!success) {
    return rc::Status(rc::INTEGRATION_TEST_DUMP_LABEL_DATA_ERROR,
                      "Error.DumpLableData fail.");
  }
  return rc::Status::OK();
}

void PerceptionTest::Stop() {
  if (!end_successfully_) {
    AWARN << "Warn!!! Program doesn't end with successfully ...";
    return;
  }
  std::unique_ptr<StatisticalAnalysisProcesses> processess(
      new StatisticalAnalysisProcesses(configs_));
  processess->StartAnalyze();
  AINFO << "program exits successfullly.";
}

bool PerceptionTest::GetConfigs() {
  XMLParamReader reader(FLAGS_integration_test_config_xml);
  if (!reader.IsSucceedToLoad()) {
    AERROR << "Jenkins INFO: load xml config file \""
           << FLAGS_integration_test_config_xml << "\" failed. exiting now...";
    return false;
  }
  configs_ = reader.GetConfigs();
  if (!configs_) {
    AERROR << "Jenkins INFO: load xml config file \""
           << FLAGS_integration_test_config_xml
           << "\" failed. configs is nullptr. exiting now...";
    return false;
  }
  AINFO << "load config xml successfully. xml = "
        << FLAGS_integration_test_config_xml;
  return true;
}

}  // namespace integration_test
}  // namespace roadstar
