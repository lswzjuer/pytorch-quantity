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

#include "modules/common/canbus/can_client/socket/socket_can_client_raw.h"

#include <vector>

#include "gtest/gtest.h"

#include "modules/msgs/canbus/proto/can_card_parameter.pb.h"

namespace roadstar {
namespace common {
namespace canbus {
namespace can {

using roadstar::common::ErrorCode;

TEST(SocketCanClientRawTest, simpletest) {
  CANCardParameter param;
  param.set_can_card_brand(CANCardParameter::SOCKET_CAN);
  param.set_channel_id(CANCardParameter::CHANNEL_ID_ZERO);

  SocketCanClientRaw socket_can_client;
  EXPECT_TRUE(socket_can_client.Init(param));
  EXPECT_EQ(socket_can_client.Start(), ErrorCode::CAN_CLIENT_ERROR_BASE);
  std::vector<CanFrame> frames;
  int32_t num = 0;
  EXPECT_EQ(socket_can_client.Send(frames, &num),
            ErrorCode::CAN_CLIENT_ERROR_SEND_FAILED);
  EXPECT_EQ(socket_can_client.Receive(&frames, &num),
            ErrorCode::CAN_CLIENT_ERROR_RECV_FAILED);
  CanFrame can_frame;
  frames.push_back(can_frame);
  EXPECT_EQ(socket_can_client.SendSingleFrame(frames),
            ErrorCode::CAN_CLIENT_ERROR_SEND_FAILED);
  socket_can_client.Stop();
}

}  // namespace can
}  // namespace canbus
}  // namespace common
}  // namespace roadstar
