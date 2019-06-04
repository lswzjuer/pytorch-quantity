#ifndef MODULES_COMMON_MODULE_CONF_MODULE_UTIL_H
#define MODULES_COMMON_MODULE_CONF_MODULE_UTIL_H

#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "gflags/gflags.h"
#include "modules/common/macro.h"
#include "modules/msgs/module_conf/proto/module_conf.pb.h"

DECLARE_string(living_modules_path);
DECLARE_string(internal_living_modules_path);
DECLARE_string(module_conf_path);

namespace roadstar {
namespace common {
namespace module_conf {

class ModuleUtil {
 public:
  static std::unordered_set<std::string> GetExternalModuleNames();

  static bool IsOptional(const std::string &module_name);

  static std::vector<roadstar::module_conf::TopicConf> GetMonitoredTopics(
      const std::string &module_name);

  static roadstar::module_conf::ProcessConf GetMonitoredProcess(
      const std::string &module_name);

  static std::vector<roadstar::module_conf::ModuleConfSet::HardwareConf>
  GetHardwares();

  static std::string GetEndpoint(const std::string &module_name);

  static std::optional<std::string> GetSupportedCommand(
      const std::string &module_name, const std::string &command_name);

  static std::unordered_set</* MessageType */ int> GetPublishingTypes(
      const std::string &module_name);

  struct RemoteEndpoint {
    adapter::AdapterConfig::MessageType message_type;
    std::string module_name;
    std::string endpoint;
    bool header_only;
  };
  static std::vector<RemoteEndpoint> GetRemoteEndpoints(
      const std::string &module_name);

  static std::vector<RemoteEndpoint> GetRemoteEndpoints(
      const std::unordered_set<int> &publishing_types);

  DECLARE_SINGLETON(ModuleUtil);

 private:
  std::unordered_map<
      std::string,
      std::pair<roadstar::module_conf::LivingModuleSet::LivingModule,
                roadstar::module_conf::ModuleConfSet::ModuleConf>>
      conf_;
  std::unordered_set<std::string> external_module_names_;
  std::vector<roadstar::module_conf::ModuleConfSet::HardwareConf> hardwares_;
};

}  // namespace module_conf
}  // namespace common
}  // namespace roadstar

#endif  // MODULES_COMMON_MODULE_CONF_MODULE_UTIL_H
