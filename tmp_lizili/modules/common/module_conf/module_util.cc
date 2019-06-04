#include "modules/common/module_conf/module_util.h"

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "modules/common/util/file.h"

DEFINE_string(
    living_modules_path,
    "config/modules/common/module_conf/conf/living_modules_conf.pb.txt",
    "Path of living modules");
DEFINE_string(internal_living_modules_path,
              "config/modules/common/module_conf/conf/"
              "internal_living_modules_conf.pb.txt",
              "Path of internal living modules");
DEFINE_string(module_conf_path,
              "config/modules/common/module_conf/conf/module_conf.pb.txt",
              "Path of static module configs");

namespace roadstar {
namespace common {
namespace module_conf {

ModuleUtil::ModuleUtil() {
  roadstar::module_conf::LivingModuleSet living_modules;
  roadstar::module_conf::LivingModuleSet internal_living_modules;
  roadstar::module_conf::ModuleConfSet module_confs;
  CHECK(
      util::GetProtoFromASCIIFile(FLAGS_living_modules_path, &living_modules));
  CHECK(util::GetProtoFromASCIIFile(FLAGS_internal_living_modules_path,
                                    &internal_living_modules));
  CHECK(util::GetProtoFromASCIIFile(FLAGS_module_conf_path, &module_confs));
  for (auto living_module : living_modules.living_modules()) {
    CHECK(!living_module.name().empty());
    conf_[living_module.name()].first = living_module;
    external_module_names_.insert(living_module.name());
  }
  for (auto living_module : internal_living_modules.living_modules()) {
    CHECK(!living_module.name().empty());
    conf_[living_module.name()].first = living_module;
  }
  for (auto module : module_confs.modules()) {
    CHECK(!module.name().empty());
    if (conf_.count(module.name())) {
      conf_[module.name()].second = module;
    }
  }
  for (auto hardware : module_confs.hardware()) {
    hardwares_.push_back(hardware);
  }
  for (auto it : conf_) {
    if (it.second.first.name() != it.second.second.name()) {
      AFATAL << "living module " << it.second.first.name() << " not found in "
             << FLAGS_module_conf_path;
    }
  }
}

std::unordered_set<std::string> ModuleUtil::GetExternalModuleNames() {
  return instance()->external_module_names_;
}

bool ModuleUtil::IsOptional(const std::string &module_name) {
  AFATAL_IF(!instance()->conf_.count(module_name))
      << "Undefined module: " << module_name;
  return instance()->conf_[module_name].first.optional();
}

std::vector<roadstar::module_conf::TopicConf> ModuleUtil::GetMonitoredTopics(
    const std::string &module_name) {
  AFATAL_IF(!instance()->conf_.count(module_name))
      << "Undefined module: " << module_name;
  std::vector<roadstar::module_conf::TopicConf> ret;
  for (auto topic :
       instance()->conf_[module_name].first.monitored_topic_conf()) {
    ret.push_back(topic);
  }
  return ret;
}

roadstar::module_conf::ProcessConf ModuleUtil::GetMonitoredProcess(
    const std::string &module_name) {
  AFATAL_IF(!instance()->conf_.count(module_name))
      << "Undefined module: " << module_name;
  return instance()->conf_[module_name].second.process_conf();
}

std::vector<roadstar::module_conf::ModuleConfSet::HardwareConf>
ModuleUtil::GetHardwares() {
  return instance()->hardwares_;
}

std::string ModuleUtil::GetEndpoint(const std::string &module_name) {
  AFATAL_IF(!instance()->conf_.count(module_name))
      << "Undefined module: " << module_name;
  return instance()->conf_[module_name].first.endpoint();
}

std::optional<std::string> ModuleUtil::GetSupportedCommand(
    const std::string &module_name, const std::string &command_name) {
  if (instance()->conf_.count(module_name) == 0) return std::nullopt;
  const auto &command_map =
      instance()->conf_[module_name].second.supported_commands();
  if (command_map.count(command_name) == 0) return std::nullopt;
  return command_map.at(command_name);
}

std::unordered_set<int> ModuleUtil::GetPublishingTypes(
    const std::string &module_name) {
  AFATAL_IF(!instance()->conf_.count(module_name))
      << "Undefined module: " << module_name;
  auto module = instance()->conf_[module_name].second;
  std::unordered_set<int> ret;
  if (module.send_all()) {
    for (int type = adapter::AdapterConfig::MessageType_MIN;
         type <= adapter::AdapterConfig::MessageType_MAX; type++) {
      if (adapter::AdapterConfig::MessageType_IsValid(type)) {
        ret.insert(type);
      }
    }
    return ret;
  }

  adapter::AdapterManagerConfig configs;
  if (!util::GetProtoFromASCIIFile(module.adapter_conf_path(), &configs)) {
    AFATAL << "Invalid adapter conf: " << module.adapter_conf_path();
  }
  for (auto config : configs.config()) {
    if (config.mode() == adapter::AdapterConfig::PUBLISH_ONLY ||
        config.mode() == adapter::AdapterConfig::DUPLEX) {
      ret.insert(config.type());
    }
  }
  return ret;
}

std::vector<ModuleUtil::RemoteEndpoint> ModuleUtil::GetRemoteEndpoints(
    const std::unordered_set<int> &publishing_types) {
  std::vector<RemoteEndpoint> ret;
  for (auto it : instance()->conf_) {
    auto living_module = it.second.first;
    auto module_conf = it.second.second;
    auto name = it.first;

    if (living_module.endpoint().empty()) {
      continue;
    }

    adapter::AdapterManagerConfig configs;
    if (!module_conf.receive_all() &&
        util::GetProtoFromASCIIFile(module_conf.adapter_conf_path(),
                                    &configs)) {
      for (auto config : configs.config()) {
        if ((config.mode() == adapter::AdapterConfig::RECEIVE_ONLY ||
             config.mode() == adapter::AdapterConfig::RECEIVE_HEADER ||
             config.mode() == adapter::AdapterConfig::DUPLEX) &&
            publishing_types.count(config.type())) {
          RemoteEndpoint endpoint = {
              config.type(), name, living_module.endpoint(),
              config.mode() == adapter::AdapterConfig::RECEIVE_HEADER};
          ret.push_back(endpoint);
        }
      }
    } else {
      for (auto type : publishing_types) {
        RemoteEndpoint endpoint = {
            static_cast<adapter::AdapterConfig::MessageType>(type), name,
            living_module.endpoint(), false};
        ret.push_back(endpoint);
      }
    }
  }
  return ret;
}

std::vector<ModuleUtil::RemoteEndpoint> ModuleUtil::GetRemoteEndpoints(
    const std::string &module_name) {
  AFATAL_IF(!instance()->conf_.count(module_name))
      << "Undefined module: " << module_name;
  auto publishing_types = ModuleUtil::GetPublishingTypes(module_name);
  return GetRemoteEndpoints(publishing_types);
}

}  // namespace module_conf
}  // namespace common
}  // namespace roadstar
