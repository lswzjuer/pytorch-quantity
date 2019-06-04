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
 * @brief Defines CanReceiver class.
 */

#ifndef MODULES_CANBUS_CAN_COMM_CAN_RECEIVER_H_
#define MODULES_CANBUS_CAN_COMM_CAN_RECEIVER_H_

#include <memory>
#include <thread>
#include <vector>

#include "modules/common/canbus/can_client/can_client.h"
#include "modules/common/canbus/can_comm/message_manager.h"
#include "modules/common/log.h"
#include "modules/common/macro.h"
#include "modules/common/proto/error_code.pb.h"

/**
 * @namespace roadstar::common::canbus
 * @brief roadstar::common::canbus
 */
namespace roadstar {
namespace common {
namespace canbus {

/**
 * @class CanReceiver
 * @brief CAN receiver.
 */
template <typename SensorType>
class CanReceiver {
 public:
  /**
   * @brief Constructor.
   */
  CanReceiver() = default;

  /**
   * @brief Destructor.
   */
  virtual ~CanReceiver() = default;

  /**
   * @brief Initialize by a CAN client, message manager.
   * @param can_client The CAN client to use for receiving messages.
   * @param pt_manager The message manager which can parse and
   *        get protocol data by message id.
   * @param enable_log If log the essential information during running.
   * @return An error code indicating the status of this initialization.
   */
  ::roadstar::common::ErrorCode Init(
      ::roadstar::common::canbus::CanClient *can_client,
      MessageManager<SensorType> *pt_manager, bool enable_log);

  /**
   * @brief Get the working status of this CAN receiver.
   *        To check if it is running.
   * @return If this CAN receiver is running.
   */
  bool IsRunning() const;

  /**
   * @brief Start the CAN receiver.
   * @return The error code indicating the status of this action.
   */
  ::roadstar::common::ErrorCode Start();

  /**
   * @brief Stop the CAN receiver.
   */
  void Stop();

 private:
  void RecvThreadFunc();

  int32_t Start(bool is_blocked);

 private:
  std::unique_ptr<std::thread> thread_;
  bool is_running_ = false;
  // CanClient, MessageManager pointer life is managed by outer program
  ::roadstar::common::canbus::CanClient *can_client_ = nullptr;
  MessageManager<SensorType> *pt_manager_ = nullptr;
  bool enable_log_ = false;
  bool is_init_ = false;

  DISALLOW_COPY_AND_ASSIGN(CanReceiver);
};

template <typename SensorType>
::roadstar::common::ErrorCode CanReceiver<SensorType>::Init(
    ::roadstar::common::canbus::CanClient *can_client,
    MessageManager<SensorType> *pt_manager, bool enable_log) {
  can_client_ = can_client;
  pt_manager_ = pt_manager;
  enable_log_ = enable_log;
  if (can_client_ == nullptr) {
    AERROR << "Invalid can client.";
    return ::roadstar::common::ErrorCode::CANBUS_ERROR;
  }
  if (pt_manager_ == nullptr) {
    AERROR << "Invalid protocol manager.";
    return ::roadstar::common::ErrorCode::CANBUS_ERROR;
  }
  is_init_ = true;
  return ::roadstar::common::ErrorCode::OK;
}

template <typename SensorType>
void CanReceiver<SensorType>::RecvThreadFunc() {
  AINFO << "Can client receiver thread starts.";
  CHECK_NOTNULL(can_client_);
  CHECK_NOTNULL(pt_manager_);

  int32_t receive_error_count = 0;
  int32_t receive_none_count = 0;
  const int32_t error_count_max = 10;
  std::chrono::duration<double, std::micro> default_period{10 * 1000};

  while (IsRunning()) {
    std::vector<::roadstar::common::canbus::CanFrame> buf;
    int32_t frame_num = ::roadstar::common::canbus::kMaxCanRecvFrameLen;
    if (can_client_->Receive(&buf, &frame_num) !=
        ::roadstar::common::ErrorCode::OK) {
      AERROR_IF_EVERY(receive_error_count++ > error_count_max, error_count_max)
          << "Received " << receive_error_count << " error messages.";
      std::this_thread::sleep_for(default_period);
      continue;
    }
    receive_error_count = 0;

    if (buf.size() != static_cast<size_t>(frame_num)) {
      AERROR << "Receiver buf size[" << buf.size()
             << "] does not match can_client returned length[" << frame_num
             << "].";
    }

    if (frame_num == 0) {
      AERROR_IF_EVERY(receive_none_count++ > error_count_max, error_count_max)
          << "Received " << receive_none_count << " empty messages.";
      std::this_thread::sleep_for(default_period);
      continue;
    }
    receive_none_count = 0;

    for (const auto &frame : buf) {
      uint8_t len = frame.len;
      uint32_t uid = frame.id;
      const uint8_t *data = frame.data;
      pt_manager_->Parse(uid, data, len);
      if (enable_log_) {
        ADEBUG << "recv_can_frame#" << frame.CanFrameString();
      }
    }
    std::this_thread::yield();
  }
  AINFO << "Can client receiver thread stopped.";
}

template <typename SensorType>
bool CanReceiver<SensorType>::IsRunning() const {
  return is_running_;
}

template <typename SensorType>
::roadstar::common::ErrorCode CanReceiver<SensorType>::Start() {
  if (is_init_ == false) {
    return ::roadstar::common::ErrorCode::CANBUS_ERROR;
  }
  is_running_ = true;

  thread_.reset(new std::thread([this] { RecvThreadFunc(); }));
  if (thread_ == nullptr) {
    AERROR << "Unable to create can client receiver thread.";
    return ::roadstar::common::ErrorCode::CANBUS_ERROR;
  }
  return ::roadstar::common::ErrorCode::OK;
}

template <typename SensorType>
void CanReceiver<SensorType>::Stop() {
  if (IsRunning()) {
    AINFO << "Stopping can client receiver ...";
    is_running_ = false;
    if (thread_ != nullptr && thread_->joinable()) {
      thread_->join();
    }
    thread_.reset();
  } else {
    AINFO << "Can client receiver is not running.";
  }
  AINFO << "Can client receiver stopped [ok].";
}

}  // namespace canbus
}  // namespace common
}  // namespace roadstar

#endif  // MODULES_CANBUS_CAN_COMM_CAN_RECEIVER_H_
