#include <algorithm>
#include <functional>
#include <memory>

#include "gflags/gflags.h"
#include "modules/common/adapters/adapter_manager.h"
#include "modules/common/adapters/proto/adapter_config.pb.h"
#include "modules/common/common_gflags.h"
#include "modules/common/log.h"
#include "modules/common/message/message_service.h"
#include "modules/common/message/tools/utils.h"

DEFINE_string(type, "", "echo message type");
DEFINE_bool(hz, false, "show hz");

using roadstar::common::adapter::AdapterConfig;
using roadstar::common::adapter::AdapterManager;
using roadstar::common::adapter::AdapterManagerConfig;
using roadstar::common::message::Hertz;
using roadstar::common::message::MessageService;

int main(int argc, char *argv[]) {
  roadstar::common::InitLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  AdapterConfig::MessageType echo_type;
  if (!AdapterConfig::MessageType_Parse(FLAGS_type, &echo_type)) {
    AFATAL << "Unknow message type";
  }

  AdapterManagerConfig configs;
  for (int type = AdapterConfig::MessageType_MIN;
       type <= AdapterConfig::MessageType_MAX; type++) {
    if (AdapterConfig::MessageType_IsValid(type)) {
      auto config = configs.add_config();
      config->set_type(static_cast<AdapterConfig::MessageType>(type));
      config->set_mode(AdapterConfig::RECEIVE_ONLY);
      config->set_message_history_limit(3);
    }
  }
  AdapterManager::InitAdapters(configs);

  Hertz hz_counter(3.0);
  MessageService::Init(
      "echo_message", [&hz_counter](const AdapterConfig::MessageType type,
                                    const std::vector<unsigned char> &buffer,
                                    bool header_only) {
        if (FLAGS_type != AdapterConfig::MessageType_Name(type)) {
          return;
        }
        if (FLAGS_hz) {
          hz_counter.ReceiveOneMessage();
        } else {
          auto data = AdapterManager::GetMessageFromBuffer(type, buffer);
          std::cout << "------------------------------------------------"
                    << std::endl;
          data->PrintDebugString();
        }
        return;
      });

  while (true) {
    if (FLAGS_hz) {
      std::cout << "  " << FLAGS_type << " Frequency: " << hz_counter.GetHz()
                << "hz      \r" << std::flush;
    }
    sleep(1);
  }

  exit(0);
}
