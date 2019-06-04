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
 * @brief The class of ProtocolData
 */

#ifndef MODULES_COMMON_CANBUS_CAN_COMM_PROTOCOL_DATA_H_
#define MODULES_COMMON_CANBUS_CAN_COMM_PROTOCOL_DATA_H_

#include <cmath>
#include "modules/common/canbus/common/canbus_consts.h"

/**
 * @namespace roadstar::common::canbus
 * @brief roadstar::common::canbus
 */
namespace roadstar {
namespace common {
namespace canbus {

/**
 * @class ProtocolData
 *
 * @brief This is the base class of protocol data for sensors (such as mobileye,
 * radar and so on).
 */
template <typename SensorType>
class ProtocolData {
 public:
  /**
   * @brief construct sensor protocol data.
   */
  ProtocolData() = default;

  /**
   * @brief destruct sensor protocol data.
   */
  virtual ~ProtocolData() = default;

  /*
   * @brief get interval period for canbus messages
   * @return the interval period in us (1e-6s)
   */
  virtual uint32_t GetPeriod() const;

  /*
   * @brief get the length of sensor protocol data. The length is usually 8.
   * @return the length of sensor protocol data.
   */
  virtual int32_t GetLength() const;

  /*
   * @brief parse received data
   * @param bytes a pointer to the input bytes
   * @param length the length of the input bytes
   * @param sensor_data the parsed sensor_data
   */
  virtual void Parse(const uint8_t *bytes, int32_t length,
                     SensorType *sensor_data) const;

  /*
   * @brief update the data
   */
  virtual void UpdateData(uint8_t *data);

  /*
   * @brief reset the sensor protocol data
   */
  virtual void Reset();

  /**
   * @brief update watch dog counter
   */
  virtual void UpdateWatchdogCounter(uint8_t *data);

  /*
   * @brief check if the value is in [lower, upper], if not , round it to bound
   */
  template <typename T>
  static T BoundedValue(T lower, T upper, T val);

  /**
   * @brief static function, used to calculate the checksum of input array.
   * @param input the pointer to the start position of input array
   * @param length the length of the input array
   * @return the value of checksum
   */
  static std::uint8_t CalculateCheckSum(const uint8_t *input,
                                        const uint32_t length);

 private:
  const int32_t data_length_ = ::roadstar::common::canbus::kCanbusMessageLength;
};

template <typename SensorType>
template <typename T>
T ProtocolData<SensorType>::BoundedValue(T lower, T upper, T val) {
  if (lower > upper) {
    return val;
  }
  if (val < lower) {
    return lower;
  }
  if (val > upper) {
    return upper;
  }
  return val;
}

// (SUM(input))^0xFF
template <typename SensorType>
uint8_t ProtocolData<SensorType>::CalculateCheckSum(const uint8_t *input,
                                                    const uint32_t length) {
  uint8_t sum = 0;
  for (std::size_t i = 0; i < length; ++i) {
    sum += input[i];
  }
  return sum ^ 0xFF;
}

template <typename SensorType>
uint32_t ProtocolData<SensorType>::GetPeriod() const {
  const uint32_t const_period = 100 * 1000;
  return const_period;
}

template <typename SensorType>
void ProtocolData<SensorType>::Parse(const uint8_t *bytes, int32_t length,
                                     SensorType *sensor_data) const {}

template <typename SensorType>
void ProtocolData<SensorType>::UpdateData(uint8_t *data) {}

template <typename SensorType>
void ProtocolData<SensorType>::UpdateWatchdogCounter(uint8_t *data) {}

template <typename SensorType>
void ProtocolData<SensorType>::Reset() {}

template <typename SensorType>
int32_t ProtocolData<SensorType>::GetLength() const {
  return data_length_;
}

}  // namespace canbus
}  // namespace common
}  // namespace roadstar

#endif  //// MODULES_COMMON_CANBUS_CAN_COMM_PROTOCOL_DATA_H_
