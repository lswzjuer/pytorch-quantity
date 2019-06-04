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
 * @file kvaser_can_client.cpp
 * @brief the encapsulate call the api of kvaser can card according to
 *can_client.h
 *interface
 **/

#include "modules/common/canbus/can_client/kvaser/kvaser_can_client.h"

namespace roadstar {
namespace common {
namespace canbus {
namespace can {

using roadstar::common::ErrorCode;

void KvaserCanClient::ErrorExit(const std::string id, canStatus stat) {
  if (stat != canOK) {
    char buf[50];
    buf[0] = '\0';
    canGetErrorText(stat, buf, sizeof(buf));
    AERROR << id << " " << stat << " " << buf;
  }
}

std::string KvaserCanClient::GetErrorString(const int32_t status) {
  return "";
}

bool KvaserCanClient::Init(const CANCardParameter &parameter) {
  if (!parameter.has_channel_id()) {
    AERROR << "Init CAN failed: parameter does not have channel id. The "
              "parameter is "
           << parameter.DebugString();
    return false;
  }
  channel_ = parameter.channel_id();

  // set bitrate
  int bitrate = 500;
  if (parameter.has_bitrate()) {
    bitrate = parameter.bitrate();
  }
  switch (bitrate) {
    case 1000:
      bitrate_ = canBITRATE_1M;
      break;
    case 500:
      bitrate_ = canBITRATE_500K;
      break;
    case 250:
      bitrate_ = canBITRATE_250K;
      break;
    case 125:
      bitrate_ = canBITRATE_125K;
      break;
    default:
      bitrate_ = canBITRATE_500K;
  }

  // set virtual can
  can_open_flags_ = canOPEN_EXCLUSIVE;

  if (parameter.has_use_extended_frame()) {
    if (parameter.use_extended_frame()) {
      can_open_flags_ = can_open_flags_ | canOPEN_REQUIRE_EXTENDED;
    }
  }

  if (parameter.has_use_virtual_can()) {
    if (parameter.use_virtual_can()) {
      can_open_flags_ = can_open_flags_ | canOPEN_ACCEPT_VIRTUAL;
    }
  }

  // init can lib
  canInitializeLibrary();
  return true;
}

ErrorCode KvaserCanClient::Start() {
  if (is_started_) {
    return ErrorCode::OK;
  }
  canStatus stat;

  // open channel
  dev_handler_ = canOpenChannel(channel_, static_cast<int>(can_open_flags_));
  if (dev_handler_ < 0) {
    ErrorExit("canOpenChannel", static_cast<canStatus>(dev_handler_));
    return ErrorCode::CAN_CLIENT_ERROR_BASE;
  }

  if (bitrate_ != 0) {
    stat = canSetBusParams(dev_handler_, bitrate_, 0, 0, 0, 0, 0);
    if (stat < 0) {
      ErrorExit("canSetBusParams", stat);
      return ErrorCode::CAN_CLIENT_ERROR_BASE;
    }
  }

  // open can bus
  stat = canBusOn(dev_handler_);
  if (stat < 0) {
    ErrorExit("canBusOn", stat);
    return ErrorCode::CAN_CLIENT_ERROR_BASE;
  }

  is_started_ = true;
  return ErrorCode::OK;
}

void KvaserCanClient::Stop() {
  if (is_started_) {
    is_started_ = false;
    (void)canBusOff(dev_handler_);
    (void)canClose(dev_handler_);
  }
}

// Synchronous transmission of CAN messages
ErrorCode KvaserCanClient::Send(const std::vector<CanFrame> &frames,
                                int32_t *const frame_num) {
  CHECK_NOTNULL(frame_num);
  CHECK_EQ(frames.size(), static_cast<size_t>(*frame_num));

  if (!is_started_) {
    AERROR << "Kvaser can client has not been initiated! Please init first!";
    return ErrorCode::CAN_CLIENT_ERROR_SEND_FAILED;
  }
  for (size_t i = 0; i < frames.size() && i < kMaxCanSendFrameLen; ++i) {
    uint8_t tmp_data[8];
    std::memset(tmp_data, 0, sizeof(tmp_data));
    std::memcpy(tmp_data, frames[i].data, frames[i].len);
    canStatus stat =
        canWrite(dev_handler_, static_cast<int64_t>(frames[i].id), tmp_data,
                 static_cast<unsigned int>(frames[i].len), 0);
    if (stat == canOK) {
      ADEBUG << "Send raw data: " << frames[i].CanFrameString();
    } else {
      AERROR << "send message failed, error code: " << stat;
      ErrorExit("canWrite", stat);
      return ErrorCode::CAN_CLIENT_ERROR_BASE;
    }
  }

  return ErrorCode::OK;
}

// buf size must be 8 bytes, every time, we receive only one frame
ErrorCode KvaserCanClient::Receive(std::vector<CanFrame> *const frames,
                                   int32_t *const frame_num) {
  if (!is_started_) {
    AERROR << "Kvaser can client is not init! Please init first!";
    return ErrorCode::CAN_CLIENT_ERROR_RECV_FAILED;
  }

  if (*frame_num > kMaxCanRecvFrameLen || *frame_num < 0) {
    AERROR << "recv can frame num not in range[0, " << kMaxCanRecvFrameLen
           << "], frame_num:" << *frame_num;
    return ErrorCode::CAN_CLIENT_ERROR_FRAME_NUM;
  }

  *frame_num = 0;
  for (int i = 0; i < kMaxCanRecvFrameLen; i++) {
    CanFrame cf;
    uint64_t time;
    unsigned int dlc, flags;
    int64_t msg_id;
    canStatus stat =
        canRead(dev_handler_, &msg_id, cf.data, &dlc, &flags, &time);
    cf.id = msg_id;
    cf.len = dlc;
    if (stat == canOK) {
      (*frame_num) += 1;
      ADEBUG << "Recive raw data: " << cf.CanFrameString() << " " << flags
             << " " << time;
      frames->push_back(cf);
    } else if (stat == canERR_NOMSG) {
      break;
    } else {
      AERROR << "receive message failed, error code: " << stat;
      return ErrorCode::CAN_CLIENT_ERROR_BASE;
    }
  }

  return ErrorCode::OK;
}

}  // namespace can
}  // namespace canbus
}  // namespace common
}  // namespace roadstar
