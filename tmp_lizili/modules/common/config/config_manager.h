#pragma once
#include <memory>
#include <string_view>
#include <unordered_map>

namespace roadstar::common {

class BaseConfig;

class ConfigManager {
 public:
  static bool Init(std::string_view configs_path);

  template <typename Message>
  static bool Emplace();

 private:
  using ConfigsType =
      std::unordered_map<std::string, std::unique_ptr<BaseConfig>>;
  static ConfigsType &Configs();
};

}  // namespace roadstar::common
