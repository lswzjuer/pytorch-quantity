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
 */

#ifndef MODULES_COMMON_CANBUS_COMMON_CANBUS_CONSTS_H_
#define MODULES_COMMON_CANBUS_COMMON_CANBUS_CONSTS_H_

#include <cstdint>

/**
 * @namespace roadstar::common::canbus
 * @brief roadstar::common::canbus
 */
namespace roadstar {
namespace common {
namespace canbus {

const int32_t kCanFrameSize = 8;
const int32_t kMaxCanSendFrameLen = 1;
const int32_t kMaxCanRecvFrameLen = 10;

const int32_t kCanbusMessageLength = 8;  // according to ISO-11891-1
const int32_t kMaxCanPort = 3;

}  // namespace canbus
}  // namespace common
}  // namespace roadstar

#endif  // MODULES_COMMON_CANBUS_COMMON_CANBUS_CONSTS_H_
