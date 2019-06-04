#include "modules/common/config/config_manager.h"
#include <gflags/gflags.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <streambuf>
#include <string>
#include <string_view>
#include "modules/common/config/config.h"
#include "modules/common/log.h"
#include "modules/common/util/string_util.h"

namespace roadstar::common {

namespace {
constexpr std::string_view kPostfix = ".pb.txt";
std::string GetMagicComment(const std::string &first_line) {
  std::string result;
  if (auto pos = first_line.find_first_not_of('#');
      pos > 0 && pos != std::string::npos) {
    result = util::Trim(first_line.substr(pos));
  }
  return result;
}
}  // namespace

namespace fs = std::filesystem;

bool ConfigManager::Init(std::string_view configs_path) {
  if (!fs::exists(configs_path) || !fs::is_directory(configs_path)) {
    AFATAL << configs_path << " is invalid.";
  }
  for (auto config_file : fs::recursive_directory_iterator(configs_path)) {
    auto config_name = fs::relative(config_file, configs_path).string();
    if (config_file.is_directory()) continue;
    if (auto postfix_pos = config_name.length() - kPostfix.length();
        postfix_pos > config_name.length() ||
        config_name.substr(postfix_pos) != kPostfix) {
      AWARN << "File " << config_file
            << " is not end with `.pb.txt`. Skiping...";
    } else {
      auto ns = config_name.substr(0, postfix_pos);
      std::ifstream file(config_file.path());
      std::string first_line;
      std::getline(file, first_line);
      auto message_name = GetMagicComment(first_line);
      if (message_name.empty()) {
        AFATAL << "Invalid magic comment for config file: " << config_file;
      }
      if (auto config = Configs().find(message_name);
          config != Configs().end()) {
        if (!config->second->Load({std::istreambuf_iterator<char>(file),
                                   std::istreambuf_iterator<char>()},
                                  ns)) {
          AFATAL << "Loading config file " << config_file << " as message "
                 << message_name << " failed.";
        }
      } else {
        std::stringstream fatal_message;
        fatal_message << "Config message " << message_name
                      << " is not registered.\n";
        fatal_message << "Registered messages are: \n";
        for (auto &[key, config] : Configs()) {
          fatal_message << "\t" << key << "\n";
        }
        AFATAL << fatal_message.str();
      }
    }
  }
  return true;
}

auto ConfigManager::Configs() -> ConfigsType & {
  static ConfigsType configs;
  return configs;
}

}  // namespace roadstar::common
