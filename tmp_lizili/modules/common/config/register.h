#include <string>
#include <string_view>
#include <unordered_map>
#include "modules/common/config/config.h"
#include "modules/common/config/config_manager.h"
#include "modules/common/util/file.h"

constexpr std::string_view kPrefix = "::roadstar::";

template <typename Message>
bool roadstar::common::ConfigManager::Emplace() {
  Configs().insert({std::string(Config<Message>::message_name_),
                    std::unique_ptr<Config<Message>>(new Config<Message>())});
  return false;
}

#define REGISTER_CONFIG(Message)                                         \
  namespace roadstar::common {                                           \
  template <>                                                            \
  std::string_view Config<Message>::message_name_ =                      \
      std::string_view(#Message).substr(kPrefix.length());               \
  template <>                                                            \
  bool Config<Message>::init_ = ConfigManager::Emplace<Message>();       \
  template <>                                                            \
  std::unordered_map<std::string, Message> Config<Message>::messages_{}; \
  template <>                                                            \
  bool Config<Message>::Load(const std::string &config_text,             \
                             const std::string &ns) {                    \
    init_ = true;                                                        \
    return roadstar::common::util::GetProtoFromString(config_text,       \
                                                      &messages_[ns]);   \
  }                                                                      \
  }
//    GetProtoFromFile(#Message, &message_);
