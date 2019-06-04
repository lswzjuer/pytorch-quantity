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

#include "modules/common/canbus/can_comm/message_manager.h"

#include <memory>
#include <set>

#include "gtest/gtest.h"

#include "modules/common/canbus/can_comm/protocol_data.h"
#include "modules/msgs/canbus/proto/chassis_detail.pb.h"

namespace roadstar {
namespace common {
namespace canbus {

using roadstar::common::ErrorCode;

class MockProtocolData
    : public ProtocolData<::roadstar::canbus::ChassisDetail> {
 public:
  static const int32_t ID = 0x111;
  MockProtocolData() {}
};

class MockMessageManager
    : public MessageManager<::roadstar::canbus::ChassisDetail> {
 public:
  MockMessageManager() {
    AddRecvProtocolData<MockProtocolData, true>();
    AddSendProtocolData<MockProtocolData, true>();
  }
};

TEST(MessageManagerTest, GetMutableProtocolDataById) {
  uint8_t mock_data = 1;
  MockMessageManager manager;
  manager.Parse(MockProtocolData::ID, &mock_data, 8);
  manager.ResetSendMessages();
  EXPECT_TRUE(manager.GetMutableProtocolDataById(MockProtocolData::ID) !=
              nullptr);

  ::roadstar::canbus::ChassisDetail chassis_detail;
  EXPECT_EQ(manager.GetSensorData(&chassis_detail), ErrorCode::OK);
  EXPECT_EQ(manager.GetSensorData(nullptr), ErrorCode::CANBUS_ERROR);
}

}  // namespace canbus
}  // namespace common
}  // namespace roadstar
