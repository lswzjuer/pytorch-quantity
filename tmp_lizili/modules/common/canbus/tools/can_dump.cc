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
              "modules/common/canbus/conf/can_client_conf.pb.txt",
              "can client conf for client");

const int32_t kMaxCanSendFrameLen = 1;
const int32_t kMaxCanRecvFrameLen = 10;

namespace roadstar {
namespace common {
namespace canbus {

void RecvLoop(const std::unique_ptr<CanClient> &client) {
  using ::roadstar::common::ErrorCode;
  using ::roadstar::common::time::AsInt64;
  using ::roadstar::common::time::Clock;
  using ::roadstar::common::time::micros;
  AINFO << "Receive loop starting...";
  int64_t start = 0, recv_time = 0;
  int64_t recv_cnt = 0;
  std::vector<CanFrame> buf;

  bool first = true;
  while (true) {
    buf.clear();
    int32_t len = kMaxCanRecvFrameLen;
    ErrorCode ret = client->Receive(&buf, &len);
    if (len == 0) {
      continue;
    }
    if (first) {
      start = AsInt64<micros>(Clock::Now());  // NOLINT
      first = false;
    }
    if (ret != ErrorCode::OK || len == 0) {
      AINFO << "recv error:" << ret;
      continue;
    }
    for (int32_t i = 0; i < len; ++i) {
      recv_cnt++;
      AINFO << "recv_frame#" << buf[i].CanFrameString()
            << ",recv_cnt: " << recv_cnt;
    }
  }
  int64_t end = AsInt64<micros>(Clock::Now());
  recv_time = end - start;
  std::cout << "recv time:" << recv_time << std::endl;
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
  RecvLoop(client);

  return 0;
}
