/******************************************************************************
 * Copyright 2017 The roadstar Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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

DEFINE_bool(only_one_send, false, "only send test.");
DEFINE_string(can_client_conf_file_a,
              "modules/canbus/conf/can_client_conf_a.pb.txt",
              "can client conf for client a");
DEFINE_string(can_client_conf_file_b,
              "modules/canbus/conf/can_client_conf_b.pb.txt",
              "can client conf for client b");
DEFINE_int64(agent_mutual_send_frames, 1000, "Every agent send frame num");

const int32_t kMaxCanSendFrameLen = 1;
const int32_t kMaxCanRecvFrameLen = 10;

namespace roadstar {
namespace common {
namespace canbus {

struct TestCanParam {
  CANCardParameter conf;
  bool is_first_agent = false;
  int32_t recv_cnt = 0;
  int32_t recv_err_cnt = 0;
  int32_t send_cnt = 0;
  int32_t send_err_cnt = 0;
  int32_t send_lost_cnt = 0;
  int32_t send_time = 0;
  int32_t recv_time = 0;
  CanClient *can_client = nullptr;

  TestCanParam() = default;

  void print() {
    AINFO << "conf: " << conf.ShortDebugString()
          << ", total send: " << send_cnt + send_err_cnt << "/"
          << FLAGS_agent_mutual_send_frames << ", send_ok: " << send_cnt
          << " , send_err_cnt: " << send_err_cnt
          << ", send_lost_cnt: " << send_lost_cnt << ", recv_cnt: " << recv_cnt
          << ", send_time: " << send_time << ", recv_time: " << recv_time;
  }
};

class CanAgent {
 public:
  explicit CanAgent(TestCanParam *param_ptr) : param_ptr_(param_ptr) {}

  TestCanParam *param_ptr() {
    return param_ptr_;
  }

  CanAgent *other_agent() {
    return other_agent_;
  }

  bool Start() {
    thread_recv_.reset(new std::thread([this] { RecvThreadFunc(); }));
    if (thread_recv_ == nullptr) {
      AERROR << "Unable to create recv thread.";
      return false;
    }
    thread_send_.reset(new std::thread([this] { SendThreadFunc(); }));
    if (thread_send_ == nullptr) {
      AERROR << "Unable to create send thread.";
      return false;
    }
    return true;
  }

  void SendThreadFunc() {
    using common::ErrorCode;
    using common::time::AsInt64;
    using common::time::Clock;
    using common::time::micros;
    AINFO << "Send thread starting...";
    TestCanParam *param = param_ptr();
    CanClient *client = param->can_client;
    std::vector<CanFrame> frames;
    frames.resize(kMaxCanSendFrameLen);

    int32_t count = 0;
    int32_t start_id = 0;
    int32_t end_id = 0;
    int32_t id = 0;
    if (param->is_first_agent) {
      start_id = 1;
      end_id = 128;
    } else {
      start_id = 129;
      end_id = start_id + 127;
    }
    id = start_id;
    int32_t send_id = id;
    AINFO << "port:" << param->conf.ShortDebugString()
          << ", start_id:" << start_id << ", end_id:" << end_id;

    // wait for other agent receiving is ok.
    while (!other_agent()->is_receiving()) {
      std::this_thread::yield();
    }
    int64_t start = AsInt64<micros>(Clock::Now());
    while (true) {
      // param->print();
      if (count >= FLAGS_agent_mutual_send_frames) {
        break;
      }
      for (int32_t i = 0; i < kMaxCanSendFrameLen; ++i) {
        // frames[i].id = id_count & 0x3FF;
        send_id = id;
        frames[i].id = id;
        frames[i].len = 8;
        frames[i].data[7] = count % 256;
        for (int32_t j = 0; j < 7; ++j) {
          frames[i].data[j] = j;
        }
        ++count;
        ++id;
        if (id > end_id) {
          id = start_id;
        }
      }
      int32_t frame_num = kMaxCanSendFrameLen;
      if (client->Send(frames, &frame_num) != ErrorCode::OK) {
        param->send_err_cnt += kMaxCanSendFrameLen;
        AERROR << "send_thread send msg failed!, id:" << send_id
               << ", conf:" << param->conf.ShortDebugString();
      } else {
        param->send_cnt += frame_num;
        param->send_lost_cnt += kMaxCanSendFrameLen - frame_num;
        AINFO << "send_frames: " << frame_num << "send_frame#"
              << frames[0].CanFrameString()
              << " send lost:" << kMaxCanSendFrameLen - frame_num
              << ", conf:" << param->conf.ShortDebugString();
      }
    }
    int64_t end = AsInt64<micros>(Clock::Now());
    param->send_time = end - start;
    // In case for finish too quick to receiver miss some msg
    sleep(2);
    AINFO << "Send thread stopping..." << param->conf.ShortDebugString();
    is_sending_finish(true);
    return;
  }

  void AddOtherAgent(CanAgent *agent) {
    other_agent_ = agent;
  }

  bool is_receiving() const {
    return is_receiving_;
  }

  void is_receiving(bool val) {
    is_receiving_ = val;
  }

  bool is_sending_finish() const {
    return is_sending_finish_;
  }

  void is_sending_finish(bool val) {
    is_sending_finish_ = val;
  }

  void RecvThreadFunc() {
    using common::ErrorCode;
    using common::time::AsInt64;
    using common::time::Clock;
    using common::time::micros;
    AINFO << "Receive thread starting...";
    TestCanParam *param = param_ptr();
    CanClient *client = param->can_client;
    int64_t start = 0;
    std::vector<CanFrame> buf;

    bool first = true;
    while (!other_agent()->is_sending_finish()) {
      is_receiving(true);
      int32_t len = kMaxCanRecvFrameLen;
      ErrorCode ret = client->Receive(&buf, &len);
      if (len == 0) {
        AINFO << "recv frame:0";
        continue;
      }
      if (first) {
        start = AsInt64<micros>(Clock::Now());
        first = false;
      }
      if (ret != ErrorCode::OK || len == 0) {
        // AINFO << "channel:" << param->conf.channel_id()
        //      << ", recv frame:failed, code:" << ret;
        AINFO << "recv error:" << ret;
        continue;
      }
      for (int32_t i = 0; i < len; ++i) {
        param->recv_cnt = param->recv_cnt + 1;
        AINFO << "recv_frame#" << buf[i].CanFrameString()
              << " conf:" << param->conf.ShortDebugString()
              << ",recv_cnt: " << param->recv_cnt;
      }
    }
    int64_t end = AsInt64<micros>(Clock::Now());
    param->recv_time = end - start;
    AINFO << "Recv thread stopping..., conf:" << param->conf.ShortDebugString();
    return;
  }

  void WaitForFinish() {
    if (thread_send_ != nullptr && thread_send_->joinable()) {
      thread_send_->join();
      thread_send_.reset();
      AINFO << "Send thread stopped. conf:"
            << param_ptr_->conf.ShortDebugString();
    }
    if (thread_recv_ != nullptr && thread_recv_->joinable()) {
      thread_recv_->join();
      thread_recv_.reset();
      AINFO << "Recv thread stopped. conf:"
            << param_ptr_->conf.ShortDebugString();
    }
  }

 private:
  bool is_receiving_ = false;
  bool is_sending_finish_ = false;
  CanAgent *other_agent_ = nullptr;
  TestCanParam *param_ptr_ = nullptr;
  std::unique_ptr<std::thread> thread_recv_;
  std::unique_ptr<std::thread> thread_send_;
};

}  // namespace canbus
}  // namespace common
}  // namespace roadstar

int main(int32_t argc, char **argv) {
  roadstar::common::InitLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  using roadstar::common::ErrorCode;
  using roadstar::common::canbus::CanAgent;
  using roadstar::common::canbus::CANCardParameter;
  using roadstar::common::canbus::CanClient;
  using roadstar::common::canbus::CanClientFactory;
  using roadstar::common::canbus::TestCanParam;
  CANCardParameter can_client_conf_a;
  std::shared_ptr<TestCanParam> param_ptr_a(new TestCanParam());
  std::shared_ptr<TestCanParam> param_ptr_b(new TestCanParam());

  auto *can_client_factory = CanClientFactory::instance();
  can_client_factory->RegisterCanClients();

  if (!roadstar::common::util::GetProtoFromFile(FLAGS_can_client_conf_file_a,
                                                &can_client_conf_a)) {
    AERROR << "Unable to load canbus conf file: "
           << FLAGS_can_client_conf_file_a;
    return 1;
  } else {
    AINFO << "Conf file is loaded: " << FLAGS_can_client_conf_file_a;
  }
  AINFO << can_client_conf_a.ShortDebugString();
  auto client_a =
      can_client_factory->CreateObject(can_client_conf_a.can_card_brand());
  if (!client_a || !client_a->Init(can_client_conf_a) ||
      client_a->Start() != ErrorCode::OK) {
    AERROR << "Create can client a failed.";
    return 1;
  }
  param_ptr_a->can_client = client_a.get();
  param_ptr_a->is_first_agent = true;
  param_ptr_a->conf = can_client_conf_a;

  CANCardParameter can_client_conf_b;
  std::unique_ptr<CanClient> client_b;
  if (!FLAGS_only_one_send) {
    if (!roadstar::common::util::GetProtoFromFile(FLAGS_can_client_conf_file_b,
                                                  &can_client_conf_b)) {
      AERROR << "Unable to load canbus conf file: "
             << FLAGS_can_client_conf_file_b;
      return 1;
    } else {
      AINFO << "Conf file is loaded: " << FLAGS_can_client_conf_file_b;
    }
    AINFO << can_client_conf_b.ShortDebugString();
    client_b =
        can_client_factory->CreateObject(can_client_conf_b.can_card_brand());
    if (!client_b || !client_b->Init(can_client_conf_b) ||
        client_b->Start() != ErrorCode::OK) {
      AERROR << "Create can client b failed.";
      return 1;
    }
    param_ptr_b->can_client = client_b.get();
    param_ptr_b->conf = can_client_conf_b;
  }

  CanAgent agent_a(param_ptr_a.get());
  CanAgent agent_b(param_ptr_b.get());
  agent_a.AddOtherAgent(&agent_b);
  agent_b.AddOtherAgent(&agent_a);
  if (!agent_a.Start()) {
    AERROR << "Agent a start failed.";
    return -1;
  }
  if (FLAGS_only_one_send) {
    agent_b.is_receiving(true);
    agent_b.is_sending_finish(true);
  } else {
    if (!agent_b.Start()) {
      AERROR << "Agent b start failed.";
      return -1;
    }
  }

  agent_a.WaitForFinish();
  if (!FLAGS_only_one_send) {
    agent_b.WaitForFinish();
  }
  param_ptr_a->print();
  if (!FLAGS_only_one_send) {
    param_ptr_b->print();
  }

  return 0;
}
