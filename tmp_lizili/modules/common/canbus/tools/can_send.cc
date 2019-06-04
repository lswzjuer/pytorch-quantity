/******************************************************************************
 * Copyright 2017 The Roadstar Authors. All Rights Reserved.
 *****************************************************************************/

#include <cstdint>
#include <iostream>
#include <memory>
#include <thread>

#include "gflags/gflags.h"
#include "modules/common/canbus/can_client/can_client.h"
#include "modules/common/canbus/can_client/can_client_factory.h"
#include "modules/common/canbus/common/byte.h"
#include "modules/common/log.h"
#include "modules/common/proto/error_code.pb.h"
#include "modules/common/time/time.h"
#include "modules/common/util/factory.h"
#include "modules/common/util/file.h"
#include "modules/msgs/canbus/proto/can_card_parameter.pb.h"

DEFINE_string(can_client_conf_file,
              "modules/common/canbus/tools/can_client_conf.pb.txt",
              "can client conf for client");

const int32_t kMaxCanSendFrameLen = 1;
const int32_t kMaxCanRecvFrameLen = 10;

namespace roadstar {
namespace common {
namespace canbus {

void SendLoop(const std::unique_ptr<CanClient> &client) {
  using ::roadstar::common::ErrorCode;
  using ::roadstar::common::time::AsInt64;
  using ::roadstar::common::time::Clock;
  using ::roadstar::common::time::micros;
  AINFO << "Send loop starting...";
  std::vector<CanFrame> buf;

  std::cout << "input your send msg, the format is 'id msg'" << std::endl;
  while (true) {
    int32_t len = kMaxCanRecvFrameLen;
    int msg_id = 0;
    std::string msg;
    int32_t frame_num = 1;
    std::cin >> msg_id >> msg;
    CanFrame frame;
    frame.id = msg_id;
    frame.len = std::min(static_cast<int>(msg.length()), 8);
    for (int i = 0; i < frame.len; i++) {
      frame.data[i] = msg[i];
    }
    AINFO << "send msg:" << frame.CanFrameString() << std::endl;
    buf.clear();
    buf.push_back(frame);
    ErrorCode ret = client->Send(buf, &frame_num);
    if (ret != ErrorCode::OK || len == 0) {
      AINFO << "send error:" << ret;
      continue;
    }
  }
  return;
}

}  // namespace canbus
}  // namespace common
}  // namespace roadstar

int main(int32_t argc, char **argv) {
  roadstar::common::InitLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  using ::roadstar::common::ErrorCode;
  using ::roadstar::common::canbus::CANCardParameter;
  using ::roadstar::common::canbus::CanClient;
  using ::roadstar::common::canbus::CanClientFactory;
  CANCardParameter can_client_conf;

  auto *can_client_factory = CanClientFactory::instance();
  can_client_factory->RegisterCanClients();

  if (!::roadstar::common::util::GetProtoFromFile(FLAGS_can_client_conf_file,
                                                  &can_client_conf)) {
    AERROR << "Unable to load canbus conf file: " << FLAGS_can_client_conf_file;
    return 1;
  } else {
    AINFO << "Conf file is loaded: " << FLAGS_can_client_conf_file;
  }
  AINFO << can_client_conf.ShortDebugString();
  auto client =
      can_client_factory->CreateObject(can_client_conf.can_card_brand());
  if (!client) {
    AERROR << "Create can client failed.";
    return 1;
  }
  if (!client->Init(can_client_conf)) {
    AERROR << "Init can client failed.";
    return 1;
  }
  if (client->Start() != ErrorCode::OK) {
    AERROR << "Start can client failed.";
    return 1;
  }
  SendLoop(client);

  return 0;
}
