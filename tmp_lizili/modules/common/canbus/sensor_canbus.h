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

#ifndef MODULES_COMMON_SENSORS_CANBUS_H_
#define MODULES_COMMON_SENSORS_CANBUS_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ros/include/ros/ros.h"

#include "modules/common/adapters/adapter_manager.h"
#include "modules/common/adapters/proto/adapter_config.pb.h"
#include "modules/common/canbus/can_client/can_client.h"
#include "modules/common/canbus/can_client/can_client_factory.h"
#include "modules/common/canbus/can_comm/can_receiver.h"
#include "modules/common/canbus/can_comm/can_sender.h"
#include "modules/common/canbus/can_comm/message_manager.h"
#include "modules/common/canbus/sensor_gflags.h"
#include "modules/common/macro.h"
#include "modules/common/monitor_log/monitor_log_buffer.h"
#include "modules/common/roadstar_app.h"
#include "modules/common/time/time.h"
#include "modules/common/util/util.h"
#include "modules/msgs/canbus/proto/can_card_parameter.pb.h"
#include "modules/msgs/canbus/proto/canbus_conf.pb.h"

/**
 * @namespace roadstar::common
 * @brief roadstar::common
 */
namespace roadstar {
namespace common {

/**
 * @class SensorCanbus
 *
 * @brief template of canbus-based sensor module main class (e.g., mobileye).
 */

using roadstar::canbus::CanbusConf;
using roadstar::common::ErrorCode;
using roadstar::common::Status;
using roadstar::common::adapter::AdapterConfig;
using roadstar::common::adapter::AdapterManager;
using roadstar::common::canbus::CanClient;
using roadstar::common::canbus::CanClientFactory;
using roadstar::common::canbus::CanReceiver;
using roadstar::common::canbus::MessageManager;
using roadstar::common::monitor::MonitorMessageItem;
using roadstar::common::time::Clock;

template <typename SensorType>
class SensorCanbus : public roadstar::common::RoadstarApp {
 public:
  SensorCanbus()
      : monitor_logger_(roadstar::common::monitor::MonitorMessageItem::CANBUS) {
  }
  ~SensorCanbus();

  /**
   * @brief obtain module name
   * @return module name
   */
  std::string Name() const override;

  /**
   * @brief module initialization function
   * @return initialization status
   */
  roadstar::common::Status Init() override;

  /**
   * @brief module start function
   * @return start status
   */
  roadstar::common::Status Start() override;

  /**
   * @brief module stop function
   */
  void Stop() override;

 private:
  void PublishSensorData();
  void OnTimer(const ros::TimerEvent &event);
  roadstar::common::Status OnError(const std::string &error_msg);
  void RegisterCanClients();

  CanbusConf canbus_conf_;
  std::unique_ptr<CanClient> can_client_;
  CanReceiver<SensorType> can_receiver_;
  std::unique_ptr<MessageManager<SensorType>> message_manager_;

  int64_t last_timestamp_ = 0;
  ros::Timer timer_;
  roadstar::common::monitor::MonitorLogger monitor_logger_;
};

// method implementations

template <typename SensorType>
std::string SensorCanbus<SensorType>::Name() const {
  return "sensor_canbus";
}

template <typename SensorType>
Status SensorCanbus<SensorType>::Init() {
  // load conf
  if (!::roadstar::common::util::GetProtoFromFile(FLAGS_sensor_conf_file,
                                                  &canbus_conf_)) {
    return OnError("Unable to load canbus conf file: " +
                   FLAGS_sensor_conf_file);
  }

  AINFO << "The canbus conf file is loaded: " << FLAGS_sensor_conf_file;
  ADEBUG << "Canbus_conf:" << canbus_conf_.ShortDebugString();

  // Init can client
  auto *can_factory = CanClientFactory::instance();
  can_factory->RegisterCanClients();
  can_client_ = can_factory->CreateCANClient(canbus_conf_.can_card_parameter());
  if (!can_client_) {
    return OnError("Failed to create can client.");
  }
  AINFO << "Can client is successfully created.";

  message_manager_ = std::unique_ptr<MessageManager<SensorType>>(
      new MessageManager<SensorType>());
  if (message_manager_ == nullptr) {
    return OnError("Failed to create message manager.");
  }
  AINFO << "Sensor message manager is successfully created.";

  if (can_receiver_.Init(can_client_.get(), message_manager_.get(),
                         canbus_conf_.enable_receiver_log()) != ErrorCode::OK) {
    return OnError("Failed to init can receiver.");
  }
  AINFO << "The can receiver is successfully initialized.";

  AdapterManager::Init(FLAGS_adapter_config_filename);

  AINFO << "The adapter manager is successfully initialized.";

  return Status::OK();
}

template <typename SensorType>
Status SensorCanbus<SensorType>::Start() {
  // 1. init and start the can card hardware
  if (can_client_->Start() != ErrorCode::OK) {
    return OnError("Failed to start can client");
  }
  AINFO << "Can client is started.";

  // 2. start receive first then send
  if (can_receiver_.Start() != ErrorCode::OK) {
    return OnError("Failed to start can receiver.");
  }
  AINFO << "Can receiver is started.";

  // 3. set timer to triger publish info periodly
  // if sensor_freq == 0, then it is event-triggered publishment.
  // no need for timer.
  if (FLAGS_sensor_freq > 0) {
    const double duration = 1.0 / FLAGS_sensor_freq;
    timer_ = AdapterManager::CreateTimer(
        ros::Duration(duration), &SensorCanbus<SensorType>::OnTimer, this);
  }

  return Status::OK();
}

template <typename SensorType>
void SensorCanbus<SensorType>::OnTimer(const ros::TimerEvent &) {
  PublishSensorData();
}

template <typename SensorType>
void SensorCanbus<SensorType>::Stop() {
  timer_.stop();

  can_receiver_.Stop();
  can_client_->Stop();
}

// Send the error to monitor and return it
template <typename SensorType>
Status SensorCanbus<SensorType>::OnError(const std::string &error_msg) {
  roadstar::common::monitor::MonitorLogBuffer buffer(&monitor_logger_);
  buffer.ERROR(error_msg);
  return Status(ErrorCode::CANBUS_ERROR, error_msg);
}

}  // namespace common
}  // namespace roadstar

#endif  // MODULES_COMMON_SENSOR_CANBUS_H_
