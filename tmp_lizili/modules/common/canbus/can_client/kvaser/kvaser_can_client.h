/******************************************************************************
 * Copyright 2017 The Roadstar Authors. All Rights Reserved.
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
 * @brief Defines the KvaserCanClient class which inherites CanClient.
 */

#ifndef MODULES_CANBUS_CAN_CLIENT_CLIENT_KVASER_CAN_CLIENT_H_
#define MODULES_CANBUS_CAN_CLIENT_CLIENT_KVASER_CAN_CLIENT_H_

#include <canlib.h>

#include <string>
#include <vector>
#include "gflags/gflags.h"
#include "modules/common/canbus/can_client/can_client.h"
#include "modules/common/canbus/common/canbus_consts.h"
#include "modules/common/proto/error_code.pb.h"
#include "modules/msgs/canbus/proto/can_card_parameter.pb.h"

/**
 * @namespace roadstar::canbus::can
 * @brief roadstar::canbus::can
 */
namespace roadstar {
namespace common {
namespace canbus {
namespace can {

/**
 * @class KvaserCanClient
 * @brief The class which defines a KVASER CAN client which inherites CanClient.
 */
class KvaserCanClient : public CanClient {
 public:
  /**
   * @brief Initialize the KVASER CAN client by specified CAN card parameters.
   * @param parameter CAN card parameters to initialize the CAN client.
   * @return If the initialization is successful.
   */
  bool Init(const CANCardParameter &parameter) override;

  /**
   * @brief Destructor
   */
  virtual ~KvaserCanClient() = default;

  /**
   * @brief Start the KVASER CAN client.
   * @return The status of the start action which is defined by
   *         roadstar::common::ErrorCode.
   */
  roadstar::common::ErrorCode Start() override;

  /**
   * @brief Stop the KVASER CAN client.
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
  roadstar::common::ErrorCode Receive(std::vector<CanFrame> *const frames,
                                      int32_t *const frame_num) override;

  /**
   * @brief Get the error string.
   * @param status The status to get the error string.
   */
  void ErrorExit(const std::string id, canStatus stat);
  std::string GetErrorString(const int32_t status) override;

 private:
  canHandle dev_handler_;
  int64_t bitrate_;
  unsigned int can_open_flags_;
  CANCardParameter::CANChannelId channel_;
};

}  // namespace can
}  // namespace canbus
}  // namespace common
}  // namespace roadstar

#endif  // MODULES_CANBUS_CANCARD_CLIENT_KVASER_CAN_CLIENT_H_
