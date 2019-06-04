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

#include "modules/common/adapters/adapter_manager.h"

#include "modules/common/adapters/adapter_gflags.h"
#include "modules/common/log.h"
#include "modules/common/message/message_service.h"
#include "modules/common/util/util.h"

namespace roadstar {
namespace common {
namespace adapter {

AdapterManager::AdapterManager() {}

void AdapterManager::Observe() {
  for (const auto observe : instance()->observers_) {
    observe();
  }
}

bool AdapterManager::Initialized() {
  return instance()->initialized_;
}

void AdapterManager::Init(const std::string &adapter_config_filename) {
  // Parse config file
  AdapterManagerConfig configs;
  CHECK(roadstar::common::util::GetProtoFromFile(adapter_config_filename,
                                                 &configs))
      << "Unable to parse adapter config file " << adapter_config_filename;
  AINFO << "Init AdapterManger config:" << configs.DebugString();
  Init(configs);
}

void AdapterManager::Init(const AdapterManagerConfig &configs) {
  instance()->initialized_ = true;
  if (configs.is_ros()) {
    instance()->node_handle_.reset(new ros::NodeHandle());
  }

  std::string name = configs.module_name();
  if (FLAGS_enable_message_service) {
    if (!name.empty()) {
      message::MessageService::Init(name,
                                    AdapterManager::MessageServiceCallback);
    } else {
      AFATAL << "adapter.conf missing module_name";
    }
  }
  InitAdapters(configs);
}

void AdapterManager::InitAdapters(const AdapterManagerConfig &configs) {
  for (const auto &config : configs.config()) {
    switch (config.type()) {
      // ins
      case AdapterConfig::INS:
        EnableIns(FLAGS_ins_topic, config.type(), config.mode(),
                  config.message_history_limit());
        break;
      // localization
      case AdapterConfig::LOCALIZATION:
        EnableLocalization(FLAGS_localization_topic, config.type(),
                           config.mode(), config.message_history_limit());
        break;
      // perception
      case AdapterConfig::CAMERA_HEAD_LEFT:
        EnableCameraHeadLeft(FLAGS_camera_head_left_topic, config.type(),
                             config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::CAMERA_HEAD_RIGHT:
        EnableCameraHeadRight(FLAGS_camera_head_right_topic, config.type(),
                              config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::CAMERA_FRONT_LEFT:
        EnableCameraFrontLeft(FLAGS_camera_front_left_topic, config.type(),
                              config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::CAMERA_FRONT_RIGHT:
        EnableCameraFrontRight(FLAGS_camera_front_right_topic, config.type(),
                               config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::CAMERA_MID_LEFT:
        EnableCameraMidLeft(FLAGS_camera_mid_left_topic, config.type(),
                            config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::CAMERA_MID_RIGHT:
        EnableCameraMidRight(FLAGS_camera_mid_right_topic, config.type(),
                             config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::CAMERA_TAIL_LEFT:
        EnableCameraTailLeft(FLAGS_camera_tail_left_topic, config.type(),
                             config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::CAMERA_TAIL_RIGHT:
        EnableCameraTailRight(FLAGS_camera_tail_right_topic, config.type(),
                              config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::CAMERA_TRAFFIC_LIGHT:
        EnableCameraTrafficLight(FLAGS_camera_traffic_light_topic,
                                 config.type(), config.mode(),
                                 config.message_history_limit());
        break;
      case AdapterConfig::CAMERA_HEAD_LEFT_PROTO:
        EnableCameraHeadLeftProto(FLAGS_camera_head_left_proto_topic,
                                  config.type(), config.mode(),
                                  config.message_history_limit());
        break;
      case AdapterConfig::CAMERA_HEAD_RIGHT_PROTO:
        EnableCameraHeadRightProto(FLAGS_camera_head_right_proto_topic,
                                   config.type(), config.mode(),
                                   config.message_history_limit());
        break;
      case AdapterConfig::CAMERA_FRONT_LEFT_PROTO:
        EnableCameraFrontLeftProto(FLAGS_camera_front_left_proto_topic,
                                   config.type(), config.mode(),
                                   config.message_history_limit());
        break;
      case AdapterConfig::CAMERA_FRONT_RIGHT_PROTO:
        EnableCameraFrontRightProto(FLAGS_camera_front_right_proto_topic,
                                    config.type(), config.mode(),
                                    config.message_history_limit());
        break;
      case AdapterConfig::CAMERA_MID_LEFT_PROTO:
        EnableCameraMidLeftProto(FLAGS_camera_mid_left_proto_topic,
                                 config.type(), config.mode(),
                                 config.message_history_limit());
        break;
      case AdapterConfig::CAMERA_MID_RIGHT_PROTO:
        EnableCameraMidRightProto(FLAGS_camera_mid_right_proto_topic,
                                  config.type(), config.mode(),
                                  config.message_history_limit());
        break;
      case AdapterConfig::CAMERA_TAIL_LEFT_PROTO:
        EnableCameraTailLeftProto(FLAGS_camera_tail_left_proto_topic,
                                  config.type(), config.mode(),
                                  config.message_history_limit());
        break;
      case AdapterConfig::CAMERA_TAIL_RIGHT_PROTO:
        EnableCameraTailRightProto(FLAGS_camera_tail_right_proto_topic,
                                   config.type(), config.mode(),
                                   config.message_history_limit());
        break;
      case AdapterConfig::CAMERA_TRAFFIC_LIGHT_PROTO:
        EnableCameraTrafficLightProto(FLAGS_camera_traffic_light_proto_topic,
                                      config.type(), config.mode(),
                                      config.message_history_limit());
        break;
      case AdapterConfig::COMPRESSED_CAMERA_HEAD_LEFT:
        EnableCompressedCameraHeadLeft(FLAGS_compressed_camera_head_left_topic,
                                       config.type(), config.mode(),
                                       config.message_history_limit());
        break;
      case AdapterConfig::COMPRESSED_CAMERA_HEAD_RIGHT:
        EnableCompressedCameraHeadRight(
            FLAGS_compressed_camera_head_right_topic, config.type(),
            config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::COMPRESSED_CAMERA_FRONT_LEFT:
        EnableCompressedCameraFrontLeft(
            FLAGS_compressed_camera_front_left_topic, config.type(),
            config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::COMPRESSED_CAMERA_FRONT_RIGHT:
        EnableCompressedCameraFrontRight(
            FLAGS_compressed_camera_front_right_topic, config.type(),
            config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::COMPRESSED_CAMERA_MID_LEFT:
        EnableCompressedCameraMidLeft(FLAGS_compressed_camera_mid_left_topic,
                                      config.type(), config.mode(),
                                      config.message_history_limit());
        break;
      case AdapterConfig::COMPRESSED_CAMERA_MID_RIGHT:
        EnableCompressedCameraMidRight(FLAGS_compressed_camera_mid_right_topic,
                                       config.type(), config.mode(),
                                       config.message_history_limit());
        break;
      case AdapterConfig::COMPRESSED_CAMERA_TAIL_LEFT:
        EnableCompressedCameraTailLeft(FLAGS_compressed_camera_tail_left_topic,
                                       config.type(), config.mode(),
                                       config.message_history_limit());
        break;
      case AdapterConfig::COMPRESSED_CAMERA_TAIL_RIGHT:
        EnableCompressedCameraTailRight(
            FLAGS_compressed_camera_tail_right_topic, config.type(),
            config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::COMPRESSED_CAMERA_TRAFFIC_LIGHT:
        EnableCompressedCameraTrafficLight(
            FLAGS_compressed_camera_traffic_light_topic, config.type(),
            config.mode(), config.message_history_limit());
        break;

      case AdapterConfig::COMPRESSED_CAMERA_HEAD_LEFT_PROTO:
        EnableCompressedCameraHeadLeftProto(
            FLAGS_compressed_camera_head_left_proto_topic, config.type(),
            config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::COMPRESSED_CAMERA_HEAD_RIGHT_PROTO:
        EnableCompressedCameraHeadRightProto(
            FLAGS_compressed_camera_head_right_proto_topic, config.type(),
            config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::COMPRESSED_CAMERA_FRONT_LEFT_PROTO:
        EnableCompressedCameraFrontLeftProto(
            FLAGS_compressed_camera_front_left_proto_topic, config.type(),
            config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::COMPRESSED_CAMERA_FRONT_RIGHT_PROTO:
        EnableCompressedCameraFrontRightProto(
            FLAGS_compressed_camera_front_right_proto_topic, config.type(),
            config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::COMPRESSED_CAMERA_MID_LEFT_PROTO:
        EnableCompressedCameraMidLeftProto(
            FLAGS_compressed_camera_mid_left_proto_topic, config.type(),
            config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::COMPRESSED_CAMERA_MID_RIGHT_PROTO:
        EnableCompressedCameraMidRightProto(
            FLAGS_compressed_camera_mid_right_proto_topic, config.type(),
            config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::COMPRESSED_CAMERA_TAIL_LEFT_PROTO:
        EnableCompressedCameraTailLeftProto(
            FLAGS_compressed_camera_tail_left_proto_topic, config.type(),
            config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::COMPRESSED_CAMERA_TAIL_RIGHT_PROTO:
        EnableCompressedCameraTailRightProto(
            FLAGS_compressed_camera_tail_right_proto_topic, config.type(),
            config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::COMPRESSED_CAMERA_TRAFFIC_LIGHT_PROTO:
        EnableCompressedCameraTrafficLightProto(
            FLAGS_compressed_camera_traffic_light_proto_topic, config.type(),
            config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::POINT_CLOUD:
        EnablePointCloud(FLAGS_point_cloud_topic, config.type(), config.mode(),
                         config.message_history_limit());
        break;
      case AdapterConfig::HESAI_POINT_CLOUD:
        EnableHESAIPointCloud(FLAGS_hesai_point_cloud_topic, config.type(),
                              config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::VLP_POINT_CLOUD1:
        EnableVLPPointCloud1(FLAGS_vlp_point_cloud1_topic, config.type(),
                             config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::VLP_POINT_CLOUD2:
        EnableVLPPointCloud2(FLAGS_vlp_point_cloud2_topic, config.type(),
                             config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::VLP_POINT_CLOUD3:
        EnableVLPPointCloud3(FLAGS_vlp_point_cloud3_topic, config.type(),
                             config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::VLP_POINT_CLOUD4:
        EnableVLPPointCloud4(FLAGS_vlp_point_cloud4_topic, config.type(),
                             config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::FUSION_MAP:
        EnableFusionMap(FLAGS_fusion_map_topic, config.type(), config.mode(),
                        config.message_history_limit());
        break;
      case AdapterConfig::RADAR_FILTER:
        EnableRadarFilter(FLAGS_radar_filter_topic, config.type(),
                          config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::TRAFFIC_LIGHT_DETECTION:
        EnableTrafficLightDetection(FLAGS_traffic_light_detection_topic,
                                    config.type(), config.mode(),
                                    config.message_history_limit());
        break;
      case AdapterConfig::LANE_DETECTION:
        EnableLaneDetection(FLAGS_lane_detection_topic, config.type(),
                            config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::VISION_LANE:
        EnableVisionLane(FLAGS_vision_lane_topic, config.type(), config.mode(),
                         config.message_history_limit());
        break;
      // esr and rsds
      case AdapterConfig::ESR:
        EnableEsr(FLAGS_esr_topic, config.type(), config.mode(),
                  config.message_history_limit());
        break;
      case AdapterConfig::RSDS:
        EnableRsds(FLAGS_rsds_topic, config.type(), config.mode(),
                   config.message_history_limit());
        break;
      // planning
      case AdapterConfig::PLANNING_TRAJECTORY:
        EnablePlanningTrajectory(FLAGS_planning_trajectory_topic, config.type(),
                                 config.mode(), config.message_history_limit());
        break;
      // control
      case AdapterConfig::CONTROL_COMMAND:
        EnableControlCommand(FLAGS_control_command_topic, config.type(),
                             config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::CONTROL_DEBUG:
        EnableControlDebug(FLAGS_control_debug_topic, config.type(),
                           config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::CONTROL_STATUS:
        EnableControlStatus(FLAGS_control_status_topic, config.type(),
                            config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::PAD:
        EnablePad(FLAGS_pad_topic, config.type(), config.mode(),
                  config.message_history_limit());
        break;
      // canbus
      case AdapterConfig::CHASSIS:
        EnableChassis(FLAGS_chassis_topic, config.type(), config.mode(),
                      config.message_history_limit());
        break;
      case AdapterConfig::CHASSIS_DETAIL:
        EnableChassisDetail(FLAGS_chassis_detail_topic, config.type(),
                            config.mode(), config.message_history_limit());
        break;
      // monitor
      case AdapterConfig::MONITOR:
        EnableMonitor(FLAGS_monitor_topic, config.type(), config.mode(),
                      config.message_history_limit());
        break;
      case AdapterConfig::SYSTEM_STATUS:
        EnableSystemStatus(FLAGS_system_status_topic, config.type(),
                           config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::DELPHIESR:
        EnableDelphiESR(FLAGS_delphi_esr_topic, config.type(), config.mode(),
                        config.message_history_limit());
        break;
      case AdapterConfig::CONTI_RADAR:
        EnableContiRadar(FLAGS_conti_radar_topic, config.type(), config.mode(),
                         config.message_history_limit());
        break;
      case AdapterConfig::CONTI_RADAR_TAIL_LEFT:
        EnableContiRadarTailLeft(FLAGS_conti_radar_tail_left_topic,
                                 config.type(), config.mode(),
                                 config.message_history_limit());
        break;
      case AdapterConfig::CONTI_RADAR_TAIL_RIGHT:
        EnableContiRadarTailRight(FLAGS_conti_radar_tail_right_topic,
                                  config.type(), config.mode(),
                                  config.message_history_limit());
        break;
      case AdapterConfig::CONTI_RADAR_HEAD_MIDDLE:
        EnableContiRadarHeadMiddle(FLAGS_conti_radar_head_middle_topic,
                                   config.type(), config.mode(),
                                   config.message_history_limit());
        break;
      case AdapterConfig::CONTI_RADAR_HEAD_LEFT:
        EnableContiRadarHeadLeft(FLAGS_conti_radar_head_left_topic,
                                 config.type(), config.mode(),
                                 config.message_history_limit());
        break;
      case AdapterConfig::CONTI_RADAR_HEAD_RIGHT:
        EnableContiRadarHeadRight(FLAGS_conti_radar_head_right_topic,
                                  config.type(), config.mode(),
                                  config.message_history_limit());
        break;
      case AdapterConfig::RAW_IMU:
        EnableRawImu(FLAGS_raw_imu_topic, config.type(), config.mode(),
                     config.message_history_limit());
        break;
      case AdapterConfig::INS_STAT:
        EnableInsStat(FLAGS_ins_stat_topic, config.type(), config.mode(),
                      config.message_history_limit());
        break;
      case AdapterConfig::INS_STATUS:
        EnableInsStatus(FLAGS_ins_status_topic, config.type(), config.mode(),
                        config.message_history_limit());
        break;
      case AdapterConfig::GNSS_STATUS:
        EnableGnssStatus(FLAGS_gnss_status_topic, config.type(), config.mode(),
                         config.message_history_limit());
        break;
      case AdapterConfig::GNSS_RAW_DATA:
        EnableGnssRawData(FLAGS_gnss_raw_data_topic, config.type(),
                          config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::GNSS_BEST_POSE:
        EnableGnssBestPose(FLAGS_gnss_best_pose_topic, config.type(),
                           config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::STREAM_STATUS:
        EnableStreamStatus(FLAGS_stream_status_topic, config.type(),
                           config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::RTCM_DATA:
        EnableRtcmData(FLAGS_rtcm_data_topic, config.type(), config.mode(),
                       config.message_history_limit());
        break;
      case AdapterConfig::PLUTO_IMU:
        EnablePlutoImu(FLAGS_pluto_imu_topic, config.type(), config.mode(),
                       config.message_history_limit());
        break;
      case AdapterConfig::ASENSING_INS:
        EnableAsensingIns(FLAGS_asensing_ins_topic, config.type(),
                          config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::ASENSING_INS_STATUS:
        EnableAsensingInsStatus(FLAGS_asensing_ins_status_topic, config.type(),
                                config.mode(), config.message_history_limit());
        break;
      // unified gnss msgs in drivers_v2
      case AdapterConfig::GNSS_INS:
        EnableGnssIns(FLAGS_gnss_ins_topic, config.type(), config.mode(),
                      config.message_history_limit());
        break;
      // perception internal msgs
      case AdapterConfig::LIDAR_PERCEPTION:
        EnableLidarPerception(FLAGS_lidar_perception_topic, config.type(),
                              config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::OLD_CAMERA_OBSTACLES:
        EnableOldCameraObstacles(FLAGS_old_camera_obstacles_topic,
                                 config.type(), config.mode(),
                                 config.message_history_limit());
        break;
      case AdapterConfig::CAMERA_OBSTACLES:
        EnableCameraObstacles(FLAGS_camera_obstacles_topic, config.type(),
                              config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::LIDAR_OBSTACLES:
        EnableLidarObstacles(FLAGS_lidar_obstacles_topic, config.type(),
                             config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::RADAR_OBSTACLES:
        EnableRadarObstacles(FLAGS_radar_obstacles_topic, config.type(),
                             config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::VELO64_PACKETS:
        EnableVelo64Packets(FLAGS_velo64_packets_topic, config.type(),
                            config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::HESAI_PACKETS:
        EnableHESAIPackets(FLAGS_hesai_packets_topic, config.type(),
                           config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::VLP1_PACKETS:
        EnableVlp1Packets(FLAGS_vlp1_packets_topic, config.type(),
                          config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::VLP2_PACKETS:
        EnableVlp2Packets(FLAGS_vlp2_packets_topic, config.type(),
                          config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::VLP3_PACKETS:
        EnableVlp3Packets(FLAGS_vlp3_packets_topic, config.type(),
                          config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::RSLIDAR_MID_LEFT:
        EnableRSLidarMidLeft(FLAGS_rslidar_mid_left_topic, config.type(),
                             config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::RSLIDAR_MID_RIGHT:
        EnableRSLidarMidRight(FLAGS_rslidar_mid_right_topic, config.type(),
                              config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::RSLIDAR_TOP_LEFT:
        EnableRSLidarTopLeft(FLAGS_rslidar_top_left_topic, config.type(),
                             config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::RSLIDAR_TOP_RIGHT:
        EnableRSLidarTopRight(FLAGS_rslidar_top_right_topic, config.type(),
                              config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::LIDAR_POINT_CLOUD_MAIN:
        EnableLidarPointCloudMain(FLAGS_lidar_point_cloud_main_topic,
                                  config.type(), config.mode(),
                                  config.message_history_limit());
        break;
      case AdapterConfig::LIDAR_POINT_CLOUD_HEAD_MID:
        EnableLidarPointCloudHeadMid(FLAGS_lidar_point_cloud_head_mid_topic,
                                     config.type(), config.mode(),
                                     config.message_history_limit());
        break;
      case AdapterConfig::LIDAR_POINT_CLOUD_TAIL_MID:
        EnableLidarPointCloudTailMid(FLAGS_lidar_point_cloud_tail_mid_topic,
                                     config.type(), config.mode(),
                                     config.message_history_limit());
        break;
      case AdapterConfig::LIDAR_POINT_CLOUD_TAIL_LEFT:
        EnableLidarPointCloudTailLeft(FLAGS_lidar_point_cloud_tail_left_topic,
                                      config.type(), config.mode(),
                                      config.message_history_limit());
        break;
      case AdapterConfig::LIDAR_POINT_CLOUD_TAIL_RIGHT:
        EnableLidarPointCloudTailRight(FLAGS_lidar_point_cloud_tail_right_topic,
                                       config.type(), config.mode(),
                                       config.message_history_limit());
        break;
      case AdapterConfig::LIDAR_POINT_CLOUD_TOP_LEFT:
        EnableLidarPointCloudTopLeft(FLAGS_lidar_point_cloud_top_left_topic,
                                     config.type(), config.mode(),
                                     config.message_history_limit());
        break;
      case AdapterConfig::LIDAR_POINT_CLOUD_TOP_RIGHT:
        EnableLidarPointCloudTopRight(FLAGS_lidar_point_cloud_top_right_topic,
                                      config.type(), config.mode(),
                                      config.message_history_limit());
        break;
      case AdapterConfig::LIDAR_POINT_CLOUD_HEAD_LEFT:
        EnableLidarPointCloudHeadLeft(FLAGS_lidar_point_cloud_head_left_topic,
                                      config.type(), config.mode(),
                                      config.message_history_limit());
        break;
      case AdapterConfig::LIDAR_POINT_CLOUD_HEAD_RIGHT:
        EnableLidarPointCloudHeadRight(FLAGS_lidar_point_cloud_head_right_topic,
                                       config.type(), config.mode(),
                                       config.message_history_limit());
        break;
      case AdapterConfig::LIDAR_PACKET_MAIN:
        EnableLidarPacketMain(FLAGS_lidar_packet_main_topic, config.type(),
                              config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::LIDAR_PACKET_HEAD_MID:
        EnableLidarPacketHeadMid(FLAGS_lidar_packet_head_mid_topic,
                                 config.type(), config.mode(),
                                 config.message_history_limit());
        break;
      case AdapterConfig::LIDAR_PACKET_TAIL_LEFT:
        EnableLidarPacketTailLeft(FLAGS_lidar_packet_tail_left_topic,
                                  config.type(), config.mode(),
                                  config.message_history_limit());
        break;
      case AdapterConfig::LIDAR_PACKET_TAIL_RIGHT:
        EnableLidarPacketTailRight(FLAGS_lidar_packet_tail_right_topic,
                                   config.type(), config.mode(),
                                   config.message_history_limit());
        break;
      case AdapterConfig::SYNC_LIDAR_HEAD:
        EnableSyncLidarHead(FLAGS_sync_lidar_head_topic, config.type(),
                            config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::SYNC_LIDAR_FRONT_LEFT:
        EnableSyncLidarFrontLeft(FLAGS_sync_lidar_front_left_topic,
                                 config.type(), config.mode(),
                                 config.message_history_limit());
        break;
      case AdapterConfig::SYNC_LIDAR_FRONT_RIGHT:
        EnableSyncLidarFrontRight(FLAGS_sync_lidar_front_right_topic,
                                  config.type(), config.mode(),
                                  config.message_history_limit());
        break;
      case AdapterConfig::SYNC_LIDAR_MID_LEFT:
        EnableSyncLidarMidLeft(FLAGS_sync_lidar_mid_left_topic, config.type(),
                               config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::SYNC_LIDAR_MID_RIGHT:
        EnableSyncLidarMidRight(FLAGS_sync_lidar_mid_right_topic, config.type(),
                                config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::SYNC_LIDAR_TAIL_LEFT:
        EnableSyncLidarTailLeft(FLAGS_sync_lidar_tail_left_topic, config.type(),
                                config.mode(), config.message_history_limit());
        break;
      case AdapterConfig::SYNC_LIDAR_TAIL_RIGHT:
        EnableSyncLidarTailRight(FLAGS_sync_lidar_tail_right_topic,
                                 config.type(), config.mode(),
                                 config.message_history_limit());
        break;
      case AdapterConfig::MESSAGE_SERVICE_STATUS:
        EnableMessageServiceStatus(FLAGS_message_service_status_topic,
                                   config.type(), config.mode(),
                                   config.message_history_limit());
        break;
      case AdapterConfig::ULTRASONIC_RADAR_FRONT:
        EnableUltrasonicRadarFront(FLAGS_ultrasonic_radar_front_topic,
                                   config.type(), config.mode(),
                                   config.message_history_limit());
        break;
      case AdapterConfig::ULTRASONIC_RADAR_MID:
        EnableUltrasonicRadarMid(FLAGS_ultrasonic_radar_mid_topic,
                                 config.type(), config.mode(),
                                 config.message_history_limit());
        break;
      case AdapterConfig::ULTRASONIC_RADAR_REAR:
        EnableUltrasonicRadarRear(FLAGS_ultrasonic_radar_rear_topic,
                                  config.type(), config.mode(),
                                  config.message_history_limit());
        break;
      default:
        AERROR << "Unknown adapter config type: "
               << AdapterConfig::MessageType_Name(config.type());
        break;
    }
  }
}

void AdapterManager::MessageServiceCallback(
    AdapterConfig::MessageType message_type,
    const std::vector<unsigned char> &buffer, bool header_only) {
  if (instance()->adapter_map_.count(message_type)) {
    if (header_only) {
      instance()->adapter_map_[message_type]->TriggerHeaderOnlyCallbacks();
    } else {
      instance()->adapter_map_[message_type]->FeedBuffer(buffer);
    }
  }
}

}  // namespace adapter
}  // namespace common
}  // namespace roadstar
