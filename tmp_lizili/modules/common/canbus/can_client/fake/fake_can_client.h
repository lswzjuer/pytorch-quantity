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

/**
 * @file
 * @brief Defines the FakeCanClient class which inherites CanClient.
 */

#ifndef MODULES_COMMON_CANBUS_CAN_CLIENT_FAKE_FAKE_CAN_CLIENT_H_
#define MODULES_COMMON_CANBUS_CAN_CLIENT_FAKE_FAKE_CAN_CLIENT_H_

#include <sstream>
#include <string>
#include <vector>

#include "modules/common/canbus/can_client/can_client.h"
#include "modules/common/proto/error_code.pb.h"

/**
 * @namespace roadstar::common::canbus::can
 * @brief roadstar::common::canbus::can
 */
namespace roadstar {
namespace common {
namespace canbus {
namespace can {

/**
 * @class FakeCanClient
 * @brief The class which defines a fake CAN client which inherits CanClient.
 *        This fake CAN client is used for testing.
 */
class FakeCanClient : public CanClient {
 public:
  /// Interval of sleeping
  static const int32_t USLEEP_INTERVAL = 10000;

  /**
   * @brief Initialize the fake CAN client by specified CAN card parameters.
   * @param parameter CAN card parameters to initialize the CAN client.
   * @return If the initialization is successful.
   */
  bool Init(const CANCardParameter &param) override;

  /**
   * @brief Destructor
   */
  virtual ~FakeCanClient() = default;

  /**
   * @brief Start the fake CAN client.
   * @return The status of the start action which is defined by
   *         roadstar::common::ErrorCode.
   */
  roadstar::common::ErrorCode Start() override;

  /**
   * @brief Stop the fake CAN client.
   */
  void Stop() override;

  /**
   * @brief Send messages
   * @param frames The messages to send.
   * @param frame_num The amount of messages to send.
   * @return The status of the sending action which is defined by
   *         roadstar::common::ErrorCode.
   */
  roadstar::common::ErrorCode Send(const std::vector<CanFrame> &frames,
                                   int32_t *const frame_num) override;

  /**
   * @brief Receive messages
   * @param frames The messages to receive.
   * @param frame_num The amount of messages to receive.
   * @return The status of the receiving action which is defined by
   *         roadstar::common::ErrorCode.
   */
  roadstar::common::ErrorCode Receive(std::vector<CanFrame> *frames,
                                      int32_t *const frame_num) override;

  /**
   * @brief Get the error string.
   * @param status The status to get the error string.
   */
  std::string GetErrorString(const int32_t status) override;

 private:
  int32_t send_counter_ = 0;
  int32_t recv_counter_ = 0;
  std::stringstream frame_info_;
};

}  // namespace can
}  // namespace canbus
}  // namespace common
}  // namespace roadstar

#endif  // MODULES_COMMON_CANBUS_CAN_CLIENT_FAKE_FAKE_CAN_CLIENT_H_
