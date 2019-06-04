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

#ifndef MODULES_ADAPTERS_MESSAGE_ADAPTERS_H_
#define MODULES_ADAPTERS_MESSAGE_ADAPTERS_H_

#include "camera_msgs/Camera.h"
#include "camera_msgs/CompressedCamera.h"
#include "camera_msgs/CompressedImageFrame.h"
#include "camera_msgs/ImageFrame.h"
#include "common_msgs/SensorSync.h"
#include "modules/common/adapters/adapter.h"
#include "modules/common/message/proto/diagnose.pb.h"
#include "modules/common/monitor_log/proto/monitor_log.pb.h"
#include "modules/msgs/canbus/proto/chassis.pb.h"
#include "modules/msgs/canbus/proto/chassis_detail.pb.h"
#include "modules/msgs/control/proto/control_command.pb.h"
#include "modules/msgs/control/proto/control_debug.pb.h"
#include "modules/msgs/control/proto/control_status.pb.h"
#include "modules/msgs/control/proto/pad_msg.pb.h"
#include "modules/msgs/drivers/asensing/proto/asensing_ins.pb.h"
#include "modules/msgs/drivers/asensing/proto/asensing_ins_status.pb.h"
#include "modules/msgs/drivers/camera/proto/image.pb.h"
#include "modules/msgs/drivers/gnss/proto/ins.pb.h"
#include "modules/msgs/drivers/lidar/proto/lidar_scan.pb.h"
#include "modules/msgs/drivers/novatel/proto/gnss.pb.h"
#include "modules/msgs/drivers/novatel/proto/gnss_best_pose.pb.h"
#include "modules/msgs/drivers/novatel/proto/gnss_status.pb.h"
#include "modules/msgs/drivers/novatel/proto/imu.pb.h"
#include "modules/msgs/drivers/novatel/proto/ins.pb.h"
#include "modules/msgs/drivers/proto/conti_radar.pb.h"
#include "modules/msgs/drivers/proto/delphi_esr.pb.h"
#include "modules/msgs/drivers/proto/esr.pb.h"
#include "modules/msgs/drivers/proto/pluto_imu.pb.h"
#include "modules/msgs/drivers/proto/rsds.pb.h"
#include "modules/msgs/drivers/radar/proto/ultrasonic_radar.pb.h"
#include "modules/msgs/localization/proto/localization.pb.h"
#include "modules/msgs/module_conf/proto/system_status.pb.h"
#include "modules/msgs/perception/proto/camera_obstacle.pb.h"
#include "modules/msgs/perception/proto/detection_result.pb.h"
#include "modules/msgs/perception/proto/fusion_map.pb.h"
#include "modules/msgs/perception/proto/lane_detection.pb.h"
#include "modules/msgs/perception/proto/obstacle_v2.pb.h"
#include "modules/msgs/perception/proto/radar_filter.pb.h"
#include "modules/msgs/perception/proto/traffic_light_detection.pb.h"
#include "modules/msgs/perception/proto/vision_lane.pb.h"
#include "modules/msgs/planning/proto/planning.pb.h"
#include "pandar_msgs/PandarScan.h"
#include "sensor_msgs/CompressedImage.h"
#include "sensor_msgs/Image.h"
#include "std_msgs/String.h"
#include "velodyne_msgs/PointCloud.h"
#include "velodyne_msgs/VelodyneScanUnified.h"

/**
 * @file message_adapters.h
 * @namespace roadstar::common::adapter
 * @brief This is an agglomeration of all the message adapters supported that
 * specializes the adapter template.
 */
namespace roadstar {
namespace common {
namespace adapter {

using CameraHeadLeftAdapter = Adapter<camera_msgs::ImageFrame>;
using CameraHeadRightAdapter = Adapter<camera_msgs::ImageFrame>;
using CameraFrontLeftAdapter = Adapter<camera_msgs::ImageFrame>;
using CameraFrontRightAdapter = Adapter<camera_msgs::ImageFrame>;
using CameraMidLeftAdapter = Adapter<camera_msgs::ImageFrame>;
using CameraMidRightAdapter = Adapter<camera_msgs::ImageFrame>;
using CameraTailLeftAdapter = Adapter<camera_msgs::ImageFrame>;
using CameraTailRightAdapter = Adapter<camera_msgs::ImageFrame>;
using CameraTrafficLightAdapter = Adapter<camera_msgs::ImageFrame>;
using ChassisAdapter = Adapter<::roadstar::canbus::Chassis>;
using ChassisDetailAdapter = Adapter<::roadstar::canbus::ChassisDetail>;

using CameraHeadLeftProtoAdapter = Adapter<roadstar::drivers::Image>;
using CameraHeadRightProtoAdapter = Adapter<roadstar::drivers::Image>;
using CameraFrontLeftProtoAdapter = Adapter<roadstar::drivers::Image>;
using CameraFrontRightProtoAdapter = Adapter<roadstar::drivers::Image>;
using CameraMidLeftProtoAdapter = Adapter<roadstar::drivers::Image>;
using CameraMidRightProtoAdapter = Adapter<roadstar::drivers::Image>;
using CameraTailLeftProtoAdapter = Adapter<roadstar::drivers::Image>;
using CameraTailRightProtoAdapter = Adapter<roadstar::drivers::Image>;
using CameraTrafficLightProtoAdapter = Adapter<roadstar::drivers::Image>;

using CompressedCameraHeadLeftAdapter =
    Adapter<camera_msgs::CompressedImageFrame>;
using CompressedCameraHeadRightAdapter =
    Adapter<camera_msgs::CompressedImageFrame>;
using CompressedCameraFrontLeftAdapter =
    Adapter<camera_msgs::CompressedImageFrame>;
using CompressedCameraFrontRightAdapter =
    Adapter<camera_msgs::CompressedImageFrame>;
using CompressedCameraMidLeftAdapter =
    Adapter<camera_msgs::CompressedImageFrame>;
using CompressedCameraMidRightAdapter =
    Adapter<camera_msgs::CompressedImageFrame>;
using CompressedCameraTailLeftAdapter =
    Adapter<camera_msgs::CompressedImageFrame>;
using CompressedCameraTailRightAdapter =
    Adapter<camera_msgs::CompressedImageFrame>;
using CompressedCameraTrafficLightAdapter =
    Adapter<camera_msgs::CompressedImageFrame>;

using CompressedCameraHeadLeftProtoAdapter =
    Adapter<roadstar::drivers::CompressedImage>;
using CompressedCameraHeadRightProtoAdapter =
    Adapter<roadstar::drivers::CompressedImage>;
using CompressedCameraFrontLeftProtoAdapter =
    Adapter<roadstar::drivers::CompressedImage>;
using CompressedCameraFrontRightProtoAdapter =
    Adapter<roadstar::drivers::CompressedImage>;
using CompressedCameraMidLeftProtoAdapter =
    Adapter<roadstar::drivers::CompressedImage>;
using CompressedCameraMidRightProtoAdapter =
    Adapter<roadstar::drivers::CompressedImage>;
using CompressedCameraTailLeftProtoAdapter =
    Adapter<roadstar::drivers::CompressedImage>;
using CompressedCameraTailRightProtoAdapter =
    Adapter<roadstar::drivers::CompressedImage>;
using CompressedCameraTrafficLightProtoAdapter =
    Adapter<roadstar::drivers::CompressedImage>;

using ControlCommandAdapter = Adapter<::roadstar::control::ControlCommand>;
using ControlDebugAdapter = Adapter<::roadstar::control::ControlDebug>;
using ControlStatusAdapter = Adapter<::roadstar::control::ControlStatus>;
using EsrAdapter = Adapter<::roadstar::drivers::Esr>;
using FusionMapAdapter = Adapter<::roadstar::perception::FusionMap>;
using InsAdapter = Adapter<::roadstar::localization::Localization>;
using LocalizationAdapter = Adapter<::roadstar::localization::Localization>;
using LaneDetectionAdapter = Adapter<::roadstar::perception::LaneDetection>;
using VisionLaneAdapter = Adapter<::roadstar::perception_v2::VisionLane>;
using MonitorAdapter = Adapter<roadstar::common::monitor::MonitorMessage>;
using PadAdapter = Adapter<::roadstar::control::PadMessage>;
using PlanningTrajectoryAdapter =
    Adapter<::roadstar::planning::PlanningTrajectory>;
using RadarFilterAdapter = Adapter<::roadstar::perception::RadarFilter>;
using RsdsAdapter = Adapter<::roadstar::drivers::Rsds>;
using SystemStatusAdapter = Adapter<roadstar::module_conf::SystemStatus>;
using TrafficLightDetectionAdapter =
    Adapter<::roadstar::perception::TrafficLightDetection>;
using VLPPointCloud1Adapter = Adapter<velodyne_msgs::PointCloud>;
using VLPPointCloud2Adapter = Adapter<velodyne_msgs::PointCloud>;
using VLPPointCloud3Adapter = Adapter<velodyne_msgs::PointCloud>;
using VLPPointCloud4Adapter = Adapter<velodyne_msgs::PointCloud>;
using HESAIPointCloudAdapter = Adapter<velodyne_msgs::PointCloud>;
using PointCloudAdapter = Adapter<velodyne_msgs::PointCloud>;
using DelphiESRAdapter = Adapter<::roadstar::drivers::DelphiESR>;
using ContiRadarAdapter = Adapter<::roadstar::drivers::ContiRadar>;
using ContiRadarTailLeftAdapter = Adapter<::roadstar::drivers::ContiRadar>;
using ContiRadarTailRightAdapter = Adapter<::roadstar::drivers::ContiRadar>;
using ContiRadarHeadMiddleAdapter = Adapter<::roadstar::drivers::ContiRadar>;
using ContiRadarHeadLeftAdapter = Adapter<::roadstar::drivers::ContiRadar>;
using ContiRadarHeadRightAdapter = Adapter<::roadstar::drivers::ContiRadar>;
using UltrasonicRadarFrontAdapter =
    Adapter<::roadstar::drivers::UltrasonicRadar>;
using UltrasonicRadarMidAdapter = Adapter<::roadstar::drivers::UltrasonicRadar>;
using UltrasonicRadarRearAdapter =
    Adapter<::roadstar::drivers::UltrasonicRadar>;
using RawImuAdapter = Adapter<::roadstar::drivers::gnss::Imu>;
using InsStatAdapter = Adapter<::roadstar::drivers::gnss::InsStat>;
using InsStatusAdapter = Adapter<::roadstar::drivers::gnss_status::InsStatus>;
using GnssStatusAdapter = Adapter<::roadstar::drivers::gnss_status::GnssStatus>;
using GnssRawDataAdapter = Adapter<std_msgs::String>;
using StreamStatusAdapter =
    Adapter<::roadstar::drivers::gnss_status::StreamStatus>;
using GnssBestPoseAdapter = Adapter<::roadstar::drivers::gnss::GnssBestPose>;
using RtcmDataAdapter = Adapter<std_msgs::String>;
using PlutoImuAdapter = Adapter<::roadstar::drivers::imu::PlutoImu>;
using AsensingInsAdapter = Adapter<::roadstar::drivers::asensing::AsensingIns>;
using AsensingInsStatusAdapter =
    Adapter<::roadstar::drivers::asensing::AsensingInsStatus>;
// unified gnss msgs in drivers_v2
using GnssInsAdapter = Adapter<::roadstar::drivers::gnss::GnssIns>;
// perception internal msgs
using LidarPerceptionAdapter =
    Adapter<::roadstar::perception::DetectionResultProto>;
using OldCameraObstaclesAdapter =
    Adapter<::roadstar::perception::CameraObstacles>;
using CameraObstaclesAdapter =
    Adapter<::roadstar::perception_v2::SensorObstacles>;
using LidarObstaclesAdapter =
    Adapter<::roadstar::perception_v2::SensorObstacles>;
using RadarObstaclesAdapter =
    Adapter<::roadstar::perception_v2::SensorObstacles>;
// perception internal msgs
using Velo64PacketsAdapter = Adapter<velodyne_msgs::VelodyneScanUnified>;
using HESAIPacketsAdapter = Adapter<pandar_msgs::PandarScan>;
using Vlp1PacketsAdapter = Adapter<velodyne_msgs::VelodyneScanUnified>;
using Vlp2PacketsAdapter = Adapter<velodyne_msgs::VelodyneScanUnified>;
using Vlp3PacketsAdapter = Adapter<velodyne_msgs::VelodyneScanUnified>;

using RSLidarMidLeftAdapter = Adapter<velodyne_msgs::PointCloud>;
using RSLidarMidRightAdapter = Adapter<velodyne_msgs::PointCloud>;
using RSLidarTopLeftAdapter = Adapter<velodyne_msgs::PointCloud>;
using RSLidarTopRightAdapter = Adapter<velodyne_msgs::PointCloud>;

using LidarPointCloudMainAdapter = Adapter<velodyne_msgs::PointCloud>;
using LidarPointCloudHeadMidAdapter = Adapter<velodyne_msgs::PointCloud>;
using LidarPointCloudTailMidAdapter = Adapter<velodyne_msgs::PointCloud>;
using LidarPointCloudTailLeftAdapter = Adapter<velodyne_msgs::PointCloud>;
using LidarPointCloudTailRightAdapter = Adapter<velodyne_msgs::PointCloud>;
using LidarPointCloudTopLeftAdapter = Adapter<velodyne_msgs::PointCloud>;
using LidarPointCloudTopRightAdapter = Adapter<velodyne_msgs::PointCloud>;
using LidarPointCloudHeadLeftAdapter = Adapter<velodyne_msgs::PointCloud>;
using LidarPointCloudHeadRightAdapter = Adapter<velodyne_msgs::PointCloud>;

using LidarPacketMainAdapter = Adapter<roadstar::drivers::lidar::LidarScan>;
using LidarPacketHeadMidAdapter = Adapter<roadstar::drivers::lidar::LidarScan>;
using LidarPacketTailLeftAdapter = Adapter<roadstar::drivers::lidar::LidarScan>;
using LidarPacketTailRightAdapter =
    Adapter<roadstar::drivers::lidar::LidarScan>;

using SyncLidarHeadAdapter = Adapter<common_msgs::SensorSync>;
using SyncLidarFrontLeftAdapter = Adapter<common_msgs::SensorSync>;
using SyncLidarFrontRightAdapter = Adapter<common_msgs::SensorSync>;
using SyncLidarMidLeftAdapter = Adapter<common_msgs::SensorSync>;
using SyncLidarMidRightAdapter = Adapter<common_msgs::SensorSync>;
using SyncLidarTailLeftAdapter = Adapter<common_msgs::SensorSync>;
using SyncLidarTailRightAdapter = Adapter<common_msgs::SensorSync>;
using MessageServiceStatusAdapter =
    Adapter<roadstar::common::message::MessageServiceStatus>;

}  // namespace adapter
}  // namespace common
}  // namespace roadstar

#endif  // MODULES_ADAPTERS_MESSAGE_ADAPTERS_H_
