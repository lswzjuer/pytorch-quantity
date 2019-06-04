#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-var-template"
#include <google/protobuf/message.h>
#include <string>
#include <type_traits>
#include <unordered_map>
#include "modules/common/log.h"

namespace roadstar::common {

class BaseConfig {
  friend class ConfigManager;

 protected:
  virtual bool Load(const std::string &config_text, const std::string &ns) {
    AERROR << "Config Not Registered";
    return false;
  };

 public:
  virtual ~BaseConfig() {}
};

template <typename Message>
class Config : public BaseConfig {
  friend class ConfigManager;

 public:
  static const Message &Get(const std::string &space) {
    if (init_) {
      if (auto message = messages_.find(space); message != messages_.end()) {
        return message->second;
      } else {
        std::stringstream fatal_message;
        fatal_message << "Cannot found config " << message_name_
                      << " for namespace " << space << "\n";
        fatal_message << "All registered namespace are:\n";
        for (auto &[key, message] : messages_) {
          fatal_message << "\t" << key << "\n";
        }
        AFATAL << fatal_message.str();
      }
    } else {
      AFATAL << "Config " << message_name_ << " not init";
    }
  }
  static auto Inited() {
    return init_;
  }
  bool Load(const std::string &config_text, const std::string &ns) override;

 protected:
 private:
  Config() {
    static_assert(std::is_base_of_v<google::protobuf::Message, Message>,
                  "Config message should be inherited from protobuf message");
  };
  static std::unordered_map<std::string, Message> messages_;
  static std::string_view message_name_;
  static bool init_;
};

}  // namespace roadstar::common

#pragma clang diagnostic pop