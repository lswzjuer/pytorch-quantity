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
 */

#ifndef MODULES_ADAPTERS_ADAPTER_MANAGER_H_
#define MODULES_ADAPTERS_ADAPTER_MANAGER_H_

#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "modules/common/adapters/adapter.h"
#include "modules/common/adapters/adapter_gflags.h"
#include "modules/common/adapters/adapter_utils.h"
#include "modules/common/adapters/message_adapters.h"
#include "modules/common/adapters/proto/adapter_config.pb.h"
#include "modules/common/log.h"
#include "modules/common/macro.h"
#include "modules/common/message/message_service.h"
#include "modules/common/transform_listener/transform_listener.h"

#include "ros/include/ros/ros.h"
#include "ros/include/rosbag/bag.h"

/**
 * @namespace roadstar::common::adapter
 * @brief roadstar::common::adapter
 */
namespace roadstar {
namespace common {
namespace adapter {

/// Macro to prepare all the necessary adapter functions when adding a
/// new input/output. For example when you want to listen to
/// car_status message for your module, you can do
/// REGISTER_ADAPTER(CarStatus) write an adapter class called
/// CarStatusAdapter, and call EnableCarStatus(`car_status_topic`,
/// true, `callback`(if there's one)) in AdapterManager.
#define REGISTER_ADAPTER(name)                                                \
 public:                                                                      \
  static void Enable##name(                                                   \
      const std::string &topic_name, AdapterConfig::MessageType type,         \
      AdapterConfig::Mode mode, int message_history_limit) {                  \
    CHECK(message_history_limit > 0)                                          \
        << "Message history limit must be greater than 0";                    \
    instance()->InternalEnable##name(topic_name, type, mode,                  \
                                     message_history_limit);                  \
  }                                                                           \
  static name##Adapter *Get##name() {                                         \
    return instance()->InternalGet##name();                                   \
  }                                                                           \
  static void Publish##name(const name##Adapter::DataType &data) {            \
    CHECK(instance()->name##_) << "Initialize adapter before publish msg";    \
    instance()->InternalPublish##name(data);                                  \
  }                                                                           \
  template <typename T>                                                       \
  static void Fill##name##Header(const std::string &module_name, T *data) {   \
    CHECK(instance()->name##_) << "Initialize adapter before filling header"; \
    static_assert(std::is_same<name##Adapter::DataType, T>::value,            \
                  "Data type must be the same with adapter's type!");         \
    instance()->name##_->FillHeader(module_name, data);                       \
  }                                                                           \
  template <typename T>                                                       \
  static void Fill##name##Header(const std::string &module_name,              \
                                 const double &sensor_time, T *data) {        \
    CHECK(instance()->name##_) << "Initialize adapter before filling header"; \
    static_assert(std::is_same<name##Adapter::DataType, T>::value,            \
                  "Data type must be the same with adapter's type!");         \
    instance()->name##_->FillHeader(module_name, sensor_time, data);          \
  }                                                                           \
  static void Add##name##Callback(name##Adapter::Callback callback) {         \
    CHECK(instance()->name##_)                                                \
        << "Initialize adapter before setting callback";                      \
    instance()->name##_->AddCallback(callback);                               \
  }                                                                           \
  static void Add##name##HeaderOnlyCallback(                                  \
      name##Adapter::HeaderOnlyCallback callback) {                           \
    CHECK(instance()->name##_)                                                \
        << "Initialize adapter before setting header_only callback";          \
    instance()->name##_->AddHeaderOnlyCallback(callback);                     \
  }                                                                           \
  template <class T>                                                          \
  static void Add##name##Callback(                                            \
      void (T::*fp)(const name##Adapter::DataType &data), T *obj) {           \
    Add##name##Callback(std::bind(fp, obj, std::placeholders::_1));           \
  }                                                                           \
  template <class T>                                                          \
  static void Add##name##HeaderOnlyCallback(                                  \
      void (T::*fp)(const name##Adapter::DataType &data), T *obj) {           \
    Add##name##HeaderOnlyCallback(std::bind(fp, obj, std::placeholders::_1)); \
  }                                                                           \
                                                                              \
 private:                                                                     \
  std::unique_ptr<name##Adapter> name##_;                                     \
  ros::Publisher name##publisher_;                                            \
  ros::Subscriber name##subscriber_;                                          \
  std::mutex name##publish_mutex_;                                            \
  AdapterConfig::MessageType name##type_;                                     \
                                                                              \
  void InternalEnable##name(                                                  \
      const std::string &topic_name, AdapterConfig::MessageType type,         \
      AdapterConfig::Mode mode, int message_history_limit) {                  \
    name##_.reset(                                                            \
        new name##Adapter(#name, topic_name, message_history_limit));         \
    adapter_map_[type] = name##_.get();                                       \
    name##type_ = type;                                                       \
    if (mode != AdapterConfig::PUBLISH_ONLY && node_handle_) {                \
      name##subscriber_ =                                                     \
          node_handle_->subscribe(topic_name, message_history_limit,          \
                                  &name##Adapter::OnReceive, name##_.get());  \
    }                                                                         \
    if (mode != AdapterConfig::RECEIVE_HEADER &&                              \
        mode != AdapterConfig::RECEIVE_ONLY && node_handle_) {                \
      name##publisher_ = node_handle_->advertise<name##Adapter::DataType>(    \
          topic_name, message_history_limit);                                 \
    }                                                                         \
    observers_.push_back([this]() { name##_->Observe(); });                   \
    PutMessageTypeToMap<name##Adapter::DataType>(type);                       \
  }                                                                           \
  name##Adapter *InternalGet##name() {                                        \
    return name##_.get();                                                     \
  }                                                                           \
  void InternalPublish##name(const name##Adapter::DataType &data) {           \
    if (FLAGS_enable_message_service && FLAGS_send_message_service) {         \
      message::MessageService::instance()->Send(name##type_, data);           \
    } else { /* Only publish ROS msg if node handle is initialized. */        \
      if (node_handle_) {                                                     \
        std::lock_guard<std::mutex> lock(name##publish_mutex_);               \
        name##publisher_.publish(data);                                       \
      }                                                                       \
    }                                                                         \
  }

/**
 * @class AdapterManager
 *
 * @brief this class hosts all the specific adapters and manages them.
 * It provides APIs for the users to initialize, access and interact
 * with the adapters that they are interested in.
 *
 * \par
 * Each (potentially) useful adapter needs to be registered here with
 * the macro REGISTER_ADAPTER.
 *
 * \par
 * The AdapterManager is a singleton.
 */
class AdapterManager {
 public:
  /**
   * @brief Initialize the /class AdapterManager singleton with the
   * provided configuration. The configuration is specified by the
   * file path.
   * @param adapter_config_filename the path to the proto file that
   * contains the adapter manager configuration.
   */
  static void Init(const std::string &adapter_config_filename);

  /**
   * @brief Initialize the /class AdapterManager singleton with the
   * provided configuration.
   * @param configs the adapter manager configuration proto.
   */
  static void Init(const AdapterManagerConfig &configs);

  static void InitAdapters(const AdapterManagerConfig &configs);

  /**
   * @brief check if the AdapterManager is initialized
   */
  static bool Initialized();

  static void Observe();

  static void WriteDataToBag(const AdapterConfig::MessageType message_type,
                             const ros::Time time,
                             const std::vector<unsigned char> &buffer,
                             rosbag::Bag *bag);

  /**
   * @brief Returns a reference to static tf2 buffer.
   */
  static tf2_ros::Buffer &Tf2Buffer() {
    static tf2_ros::Buffer tf2_buffer;
    static TransformListener tf2_listener(&tf2_buffer,
                                          instance()->node_handle_.get());
    return tf2_buffer;
  }

  /**
   * @brief create a timer which will call a callback at the specified
   * rate. It takes a class member function, and a bare pointer to the
   * object to call the method on.
   */
  template <class T>
  static ros::Timer CreateTimer(ros::Duration period,
                                void (T::*callback)(const ros::TimerEvent &),
                                T *obj, bool oneshot = false,
                                bool autostart = true) {
    CHECK(instance()->node_handle_)
        << "ROS node is only available in ROS mode, "
           "check your adapter config file!";
    return instance()->node_handle_->createTimer(period, callback, obj, oneshot,
                                                 autostart);
  }

  static std::unique_ptr<google::protobuf::Message> GetMessageFromBuffer(
      AdapterConfig::MessageType type,
      const std::vector<unsigned char> &buffer) {
    if (!instance()->message_factory_map_.count(type)) {
      AFATAL << "Invalid type " << AdapterConfig::MessageType_Name(type);
    }
    std::unique_ptr<google::protobuf::Message> data =
        instance()->message_factory_map_[type]();
    data->ParseFromArray(&buffer[0], buffer.size());
    return data;
  }

 private:
  static void MessageServiceCallback(AdapterConfig::MessageType message_type,
                                     const std::vector<unsigned char> &buffer,
                                     bool header_only);
  template <typename InputMessageType>
  typename std::enable_if<IS_PROTOBUF>::type PutMessageTypeToMap(
      AdapterConfig::MessageType type) {
    message_factory_map_[type] = []() {
      return std::make_unique<InputMessageType>();
    };
  }
  template <typename InputMessageType>
  typename std::enable_if<!IS_PROTOBUF>::type PutMessageTypeToMap(
      AdapterConfig::MessageType type) {
    return;
  }
  /// The node handler of ROS, owned by the /class AdapterManager
  /// singleton.
  std::unique_ptr<ros::NodeHandle> node_handle_;

  /// Observe() callbacks that will be used to to call the Observe()
  /// of enabled adapters.
  std::vector<std::function<void()>> observers_;

  std::unordered_map<int, AdapterBase *> adapter_map_;

  std::unordered_map<
      int, std::function<std::unique_ptr<google::protobuf::Message>()>>
      message_factory_map_;

  bool initialized_ = false;

  /// The following code registered all the adapters of interest.
  REGISTER_ADAPTER(CameraHeadLeft);
  REGISTER_ADAPTER(CameraHeadRight);
  REGISTER_ADAPTER(CameraFrontLeft);
  REGISTER_ADAPTER(CameraFrontRight);
  REGISTER_ADAPTER(CameraMidLeft);
  REGISTER_ADAPTER(CameraMidRight);
  REGISTER_ADAPTER(CameraTailLeft);
  REGISTER_ADAPTER(CameraTailRight);
  REGISTER_ADAPTER(CameraTrafficLight);
  REGISTER_ADAPTER(Chassis);
  REGISTER_ADAPTER(ChassisDetail);
  REGISTER_ADAPTER(CameraHeadLeftProto);
  REGISTER_ADAPTER(CameraHeadRightProto);
  REGISTER_ADAPTER(CameraFrontLeftProto);
  REGISTER_ADAPTER(CameraFrontRightProto);
  REGISTER_ADAPTER(CameraMidLeftProto);
  REGISTER_ADAPTER(CameraMidRightProto);
  REGISTER_ADAPTER(CameraTailLeftProto);
  REGISTER_ADAPTER(CameraTailRightProto);
  REGISTER_ADAPTER(CameraTrafficLightProto);
  REGISTER_ADAPTER(CompressedCameraHeadLeft);
  REGISTER_ADAPTER(CompressedCameraHeadRight);
  REGISTER_ADAPTER(CompressedCameraFrontLeft);
  REGISTER_ADAPTER(CompressedCameraFrontRight);
  REGISTER_ADAPTER(CompressedCameraMidLeft);
  REGISTER_ADAPTER(CompressedCameraMidRight);
  REGISTER_ADAPTER(CompressedCameraTailLeft);
  REGISTER_ADAPTER(CompressedCameraTailRight);
  REGISTER_ADAPTER(CompressedCameraTrafficLight);
  REGISTER_ADAPTER(CompressedCameraHeadLeftProto);
  REGISTER_ADAPTER(CompressedCameraHeadRightProto);
  REGISTER_ADAPTER(CompressedCameraFrontLeftProto);
  REGISTER_ADAPTER(CompressedCameraFrontRightProto);
  REGISTER_ADAPTER(CompressedCameraMidLeftProto);
  REGISTER_ADAPTER(CompressedCameraMidRightProto);
  REGISTER_ADAPTER(CompressedCameraTailLeftProto);
  REGISTER_ADAPTER(CompressedCameraTailRightProto);
  REGISTER_ADAPTER(CompressedCameraTrafficLightProto);
  REGISTER_ADAPTER(ControlCommand);
  REGISTER_ADAPTER(ControlDebug);
  REGISTER_ADAPTER(ControlStatus);
  REGISTER_ADAPTER(Esr);
  REGISTER_ADAPTER(LaneDetection);
  REGISTER_ADAPTER(VisionLane);
  REGISTER_ADAPTER(Ins);
  REGISTER_ADAPTER(Localization);
  REGISTER_ADAPTER(Monitor);
  REGISTER_ADAPTER(FusionMap);
  REGISTER_ADAPTER(Pad);
  REGISTER_ADAPTER(PointCloud);
  REGISTER_ADAPTER(HESAIPointCloud);
  REGISTER_ADAPTER(PlanningTrajectory);
  REGISTER_ADAPTER(RadarFilter);
  REGISTER_ADAPTER(Rsds);
  REGISTER_ADAPTER(SystemStatus);
  REGISTER_ADAPTER(TrafficLightDetection);
  REGISTER_ADAPTER(VLPPointCloud1);
  REGISTER_ADAPTER(VLPPointCloud2);
  REGISTER_ADAPTER(VLPPointCloud3);
  REGISTER_ADAPTER(VLPPointCloud4);
  REGISTER_ADAPTER(DelphiESR);
  REGISTER_ADAPTER(ContiRadar);
  REGISTER_ADAPTER(ContiRadarTailLeft);
  REGISTER_ADAPTER(ContiRadarTailRight);
  REGISTER_ADAPTER(ContiRadarHeadMiddle);
  REGISTER_ADAPTER(ContiRadarHeadLeft);
  REGISTER_ADAPTER(ContiRadarHeadRight);
  REGISTER_ADAPTER(UltrasonicRadarFront);
  REGISTER_ADAPTER(UltrasonicRadarMid);
  REGISTER_ADAPTER(UltrasonicRadarRear);
  REGISTER_ADAPTER(RawImu);
  REGISTER_ADAPTER(InsStat);
  REGISTER_ADAPTER(InsStatus);
  REGISTER_ADAPTER(GnssStatus);
  REGISTER_ADAPTER(GnssRawData);
  REGISTER_ADAPTER(StreamStatus);
  REGISTER_ADAPTER(GnssBestPose);
  REGISTER_ADAPTER(RtcmData);
  REGISTER_ADAPTER(PlutoImu);
  REGISTER_ADAPTER(LidarPerception);
  REGISTER_ADAPTER(OldCameraObstacles);
  REGISTER_ADAPTER(CameraObstacles);
  REGISTER_ADAPTER(LidarObstacles);
  REGISTER_ADAPTER(RadarObstacles);
  REGISTER_ADAPTER(Velo64Packets);
  REGISTER_ADAPTER(HESAIPackets);
  REGISTER_ADAPTER(Vlp1Packets);
  REGISTER_ADAPTER(Vlp2Packets);
  REGISTER_ADAPTER(Vlp3Packets);
  REGISTER_ADAPTER(AsensingIns);
  REGISTER_ADAPTER(AsensingInsStatus);
  REGISTER_ADAPTER(GnssIns);
  REGISTER_ADAPTER(RSLidarMidLeft);
  REGISTER_ADAPTER(RSLidarMidRight);
  REGISTER_ADAPTER(RSLidarTopLeft);
  REGISTER_ADAPTER(RSLidarTopRight);

  REGISTER_ADAPTER(LidarPointCloudMain);
  REGISTER_ADAPTER(LidarPointCloudHeadMid);
  REGISTER_ADAPTER(LidarPointCloudTailMid);
  REGISTER_ADAPTER(LidarPointCloudTailLeft);
  REGISTER_ADAPTER(LidarPointCloudTailRight);
  REGISTER_ADAPTER(LidarPointCloudTopLeft);
  REGISTER_ADAPTER(LidarPointCloudTopRight);
  REGISTER_ADAPTER(LidarPointCloudHeadLeft);
  REGISTER_ADAPTER(LidarPointCloudHeadRight);

  REGISTER_ADAPTER(LidarPacketMain);
  REGISTER_ADAPTER(LidarPacketHeadMid);
  REGISTER_ADAPTER(LidarPacketTailLeft);
  REGISTER_ADAPTER(LidarPacketTailRight);

  REGISTER_ADAPTER(SyncLidarHead);
  REGISTER_ADAPTER(SyncLidarFrontLeft);
  REGISTER_ADAPTER(SyncLidarFrontRight);
  REGISTER_ADAPTER(SyncLidarMidLeft);
  REGISTER_ADAPTER(SyncLidarMidRight);
  REGISTER_ADAPTER(SyncLidarTailLeft);
  REGISTER_ADAPTER(SyncLidarTailRight);

  REGISTER_ADAPTER(MessageServiceStatus);

  DECLARE_SINGLETON(AdapterManager);
};

}  // namespace adapter
}  // namespace common
}  // namespace roadstar

#endif
