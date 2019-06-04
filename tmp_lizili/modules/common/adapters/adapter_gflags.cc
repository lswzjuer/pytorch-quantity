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

#include "modules/common/adapters/adapter_gflags.h"

DEFINE_bool(enable_adapter_dump, false,
            "Whether enable dumping the messages to "
            "/tmp/adapters/<topic_name>/<seq_num>.txt for debugging purposes.");
DEFINE_bool(send_message_service, true,
            "Use MessageService instead of ROS to send message");
DEFINE_bool(enable_message_service, true, "Enable message service");
DEFINE_string(camera_head_left_topic,
              "/roadstar/drivers/pylon_camera/camera/frame/head_left/image_raw",
              "camera head_left topic name");
DEFINE_string(
    camera_head_right_topic,
    "/roadstar/drivers/pylon_camera/camera/frame/head_right/image_raw",
    "camera head_right topic name");
DEFINE_string(
    camera_front_left_topic,
    "/roadstar/drivers/pylon_camera/camera/frame/front_left/image_raw",
    "camera front_left topic name");
DEFINE_string(
    camera_front_right_topic,
    "/roadstar/drivers/pylon_camera/camera/frame/front_right/image_raw",
    "camera front_right topic name");
DEFINE_string(camera_mid_left_topic,
              "/roadstar/drivers/pylon_camera/camera/frame/mid_left/image_raw",
              "camera mid_left topic name");
DEFINE_string(camera_mid_right_topic,
              "/roadstar/drivers/pylon_camera/camera/frame/mid_right/image_raw",
              "camera mid_right topic name");
DEFINE_string(camera_tail_left_topic,
              "/roadstar/drivers/pylon_camera/camera/frame/tail_left/image_raw",
              "camera tail_left topic name");
DEFINE_string(
    camera_tail_right_topic,
    "/roadstar/drivers/pylon_camera/camera/frame/tail_right/image_raw",
    "camera tail_right topic name");
DEFINE_string(camera_traffic_light_topic,
              "/roadstar/drivers/pylon_camera/camera/frame/head_left/"
              "traffic_light/image_raw",
              "camera traffic_light topic name");

DEFINE_string(camera_head_left_proto_topic,
              "/roadstar/drivers/camera/Image/head_left",
              "camera head_left topic name");
DEFINE_string(camera_head_right_proto_topic,
              "/roadstar/drivers/camera/Image/head_right",
              "camera head_right topic name");
DEFINE_string(camera_front_left_proto_topic,
              "/roadstar/drivers/camera/Image/front_left",
              "camera front_left topic name");
DEFINE_string(camera_front_right_proto_topic,
              "/roadstar/drivers/camera/Image/front_right",
              "camera front_right topic name");
DEFINE_string(camera_mid_left_proto_topic,
              "/roadstar/drivers/camera/Image/mid_left",
              "camera mid_left topic name");
DEFINE_string(camera_mid_right_proto_topic,
              "/roadstar/drivers/camera/Image/mid_right",
              "camera mid_right topic name");
DEFINE_string(camera_tail_left_proto_topic,
              "/roadstar/drivers/camera/Image/tail_left",
              "camera tail_left topic name");
DEFINE_string(camera_tail_right_proto_topic,
              "/roadstar/drivers/camera/Image/tail_right",
              "camera tail_right topic name");
DEFINE_string(camera_traffic_light_proto_topic,
              "/roadstar/drivers/camera/Image/traffic_light",
              "camera traffic_light topic name");

DEFINE_string(chassis_topic, "/roadstar/canbus/chassis", "chassis topic name");
DEFINE_string(chassis_detail_topic, "/roadstar/canbus/chassis_detail",
              "chassis detail topic name");
DEFINE_string(compressed_camera_head_left_topic,
              "/roadstar/drivers/pylon_camera/camera/frame/head_left/jpg",
              "compressed camera head_left topic name");
DEFINE_string(compressed_camera_head_right_topic,
              "/roadstar/drivers/pylon_camera/camera/frame/head_right/jpg",
              "compressed camera head_right topic name");
DEFINE_string(compressed_camera_front_left_topic,
              "/roadstar/drivers/pylon_camera/camera/frame/front_left/jpg",
              "compressed camera front_left topic name");
DEFINE_string(compressed_camera_front_right_topic,
              "/roadstar/drivers/pylon_camera/camera/frame/front_right/jpg",
              "compressed camera front_right topic name");
DEFINE_string(compressed_camera_mid_left_topic,
              "/roadstar/drivers/pylon_camera/camera/frame/mid_left/jpg",
              "compressed camera mid_left topic name");
DEFINE_string(compressed_camera_mid_right_topic,
              "/roadstar/drivers/pylon_camera/camera/frame/mid_right/jpg",
              "compressed camera mid_right topic name");
DEFINE_string(compressed_camera_tail_left_topic,
              "/roadstar/drivers/pylon_camera/camera/frame/tail_left/jpg",
              "compressed camera tail_left topic name");
DEFINE_string(compressed_camera_tail_right_topic,
              "/roadstar/drivers/pylon_camera/camera/frame/tail_right/jpg",
              "compressed camera tail_right topic name");
DEFINE_string(
    compressed_camera_traffic_light_topic,
    "/roadstar/drivers/pylon_camera/camera/frame/head_left/traffic_light/jpg",
    "compressed camera traffic_light topic name");

DEFINE_string(compressed_camera_head_left_proto_topic,
              "/roadstar/drivers/camera/CompressedImage/head_left",
              "compressed camera head_left topic name");
DEFINE_string(compressed_camera_head_right_proto_topic,
              "/roadstar/drivers/camera/CompressedImage/head_right",
              "compressed camera head_right topic name");
DEFINE_string(compressed_camera_front_left_proto_topic,
              "/roadstar/drivers/camera/CompressedImage/front_left",
              "compressed camera front_left topic name");
DEFINE_string(compressed_camera_front_right_proto_topic,
              "/roadstar/drivers/camera/CompressedImage/front_right",
              "compressed camera front_right topic name");
DEFINE_string(compressed_camera_mid_left_proto_topic,
              "/roadstar/drivers/camera/CompressedImage/mid_left",
              "compressed camera mid_left topic name");
DEFINE_string(compressed_camera_mid_right_proto_topic,
              "/roadstar/drivers/camera/CompressedImage/mid_right",
              "compressed camera mid_right topic name");
DEFINE_string(compressed_camera_tail_left_proto_topic,
              "/roadstar/drivers/camera/CompressedImage/tail_left",
              "compressed camera tail_left topic name");
DEFINE_string(compressed_camera_tail_right_proto_topic,
              "/roadstar/drivers/camera/CompressedImage/tail_right",
              "compressed camera tail_right topic name");
DEFINE_string(compressed_camera_traffic_light_proto_topic,
              "/roadstar/drivers/camera/CompressedImage/traffic_light",
              "compressed camera traffic_light topic name");

DEFINE_string(control_command_topic, "/roadstar/control/control_command",
              "control command topic name");
DEFINE_string(control_debug_topic, "/roadstar/control/control_debug",
              "control debug topic name");
DEFINE_string(control_status_topic, "/roadstar/control/control_status",
              "control status topic name");
DEFINE_string(esr_topic, "/roadstar/drivers/esr", "delphi esr topic name");
DEFINE_string(fusion_map_topic, "/roadstar/perception/fusion_map",
              "fusion map topic name");
DEFINE_string(ins_topic, "/roadstar/drivers/ins", "ins topic name");
DEFINE_string(localization_topic, "/roadstar/localization",
              "localization topic name");
DEFINE_string(lane_detection_topic, "/roadstar/perception/lane",
              "lane detection topic name");
DEFINE_string(vision_lane_topic, "/roadstar/perception/vision_lane",
              "vision lane detection topic name");
DEFINE_string(monitor_topic, "/roadstar/monitor", "ROS topic for monitor");
DEFINE_string(pad_topic, "/roadstar/control/pad",
              "control pad message topic name");
DEFINE_string(planning_trajectory_topic, "/roadstar/planning/trajectory",
              "planning trajectory topic name");

DEFINE_string(point_cloud_topic, "/roadstar/drivers/velodyne64/PointCloud",
              "point cloud topic name ");
DEFINE_string(hesai_point_cloud_topic, "/roadstar/drivers/Pandar40/PointCloud",
              "hesai point cloud topic name ");
DEFINE_string(vlp_point_cloud1_topic,
              "/roadstar/drivers/velodyne16/PointCloud_1",
              "vlp point cloud topic name ");
DEFINE_string(vlp_point_cloud2_topic,
              "/roadstar/drivers/velodyne16/PointCloud_2",
              "vlp point cloud topic name ");
DEFINE_string(vlp_point_cloud3_topic,
              "/roadstar/drivers/velodyne16/PointCloud_3",
              "vlp point cloud topic name ");
DEFINE_string(vlp_point_cloud4_topic,
              "/roadstar/drivers/velodyne16/PointCloud_4",
              "vlp point cloud topic name ");
DEFINE_string(rslidar_mid_left_topic,
              "/roadstar/drivers/rslidar/PointCloud/mid_left",
              "rslidar mid left point cloud topic name ");
DEFINE_string(rslidar_mid_right_topic,
              "/roadstar/drivers/rslidar/PointCloud/mid_right",
              "rslidar mid right point cloud topic name ");
DEFINE_string(rslidar_top_left_topic,
              "/roadstar/drivers/rslidar/PointCloud/top_left",
              "rslidar top left point cloud topic name ");
DEFINE_string(rslidar_top_right_topic,
              "/roadstar/drivers/rslidar/PointCloud/top_right",
              "rslidar top right point cloud topic name ");

DEFINE_string(lidar_point_cloud_main_topic,
              "/roadstar/drivers/lidar/pointcloud/main",
              "lidar main point cloud topic name ");
DEFINE_string(lidar_point_cloud_head_mid_topic,
              "/roadstar/drivers/lidar/pointcloud/head_mid",
              "lidar head mid point cloud topic name ");
DEFINE_string(lidar_point_cloud_tail_mid_topic,
              "/roadstar/drivers/lidar/pointcloud/tail_mid",
              "lidar tail mid point cloud topic name ");
DEFINE_string(lidar_point_cloud_tail_left_topic,
              "/roadstar/drivers/lidar/pointcloud/tail_left",
              "lidar tail left point cloud topic name ");
DEFINE_string(lidar_point_cloud_tail_right_topic,
              "/roadstar/drivers/lidar/pointcloud/tail_right",
              "lidar tail right point cloud topic name ");
DEFINE_string(lidar_point_cloud_top_left_topic,
              "/roadstar/drivers/lidar/pointcloud/top_left",
              "lidar top left point cloud topic name ");
DEFINE_string(lidar_point_cloud_top_right_topic,
              "/roadstar/drivers/lidar/pointcloud/top_right",
              "lidar top right point cloud topic name ");
DEFINE_string(lidar_point_cloud_head_left_topic,
              "/roadstar/drivers/lidar/pointcloud/head_left",
              "lidar head left point cloud topic name ");
DEFINE_string(lidar_point_cloud_head_right_topic,
              "/roadstar/drivers/lidar/pointcloud/head_right",
              "lidar head right point cloud topic name ");

DEFINE_string(lidar_packet_main_topic, "/roadstar/drivers/lidar/packets/main",
              "lidar main point cloud topic name ");
DEFINE_string(lidar_packet_head_mid_topic,
              "/roadstar/drivers/lidar/packets/head_mid",
              "lidar front mid point cloud topic name ");
DEFINE_string(lidar_packet_tail_left_topic,
              "/roadstar/drivers/lidar/packets/tail_left",
              "lidar tail left point cloud topic name ");
DEFINE_string(lidar_packet_tail_right_topic,
              "/roadstar/drivers/lidar/packets/tail_right",
              "lidar tail right point cloud topic name ");

DEFINE_string(perception_obstacle_topic, "/roadstar/perception/obstacles",
              "perception obstacle topic name");
DEFINE_string(radar_filter_topic, "/roadstar/perception/radar_filter",
              "perception radar filter topic name");
DEFINE_string(rsds_topic, "/roadstar/drivers/rsds", "delphi rsds topic name");
DEFINE_string(system_status_topic, "/roadstar/monitor/system_status",
              "System status topic name");
DEFINE_string(traffic_light_detection_topic,
              "/roadstar/perception/traffic_light",
              "traffic light detection topic name");
DEFINE_string(delphi_esr_topic, "/roadstar/drivers/delphi_esr",
              "delphi_esr topic name");
DEFINE_string(conti_radar_topic, "/roadstar/drivers/conti_radar",
              "tail middle conti_radar topic name");
DEFINE_string(conti_radar_tail_left_topic,
              "/roadstar/drivers/conti_radar/tail_left",
              "tail left conti_radar topic name");
DEFINE_string(conti_radar_tail_right_topic,
              "/roadstar/drivers/conti_radar/tail_right",
              "tail right conti_radar topic name");
DEFINE_string(conti_radar_head_middle_topic,
              "/roadstar/drivers/conti_radar/head_mid",
              "head middle conti_radar topic name");
DEFINE_string(conti_radar_head_left_topic,
              "/roadstar/drivers/conti_radar/head_left",
              "head left conti_radar topic name");
DEFINE_string(conti_radar_head_right_topic,
              "/roadstar/drivers/conti_radar/head_right",
              "head right conti_radar topic name");
DEFINE_string(ultrasonic_radar_front_topic,
              "/roadstar/drivers/ultrasonic_radar_front",
              "ultrasonic radar topic name");
DEFINE_string(ultrasonic_radar_mid_topic,
              "/roadstar/drivers/ultrasonic_radar_mid",
              "ultrasonic radar topic name");
DEFINE_string(ultrasonic_radar_rear_topic,
              "/roadstar/drivers/ultrasonic_radar_rear",
              "ultrasonic radar topic name");
DEFINE_string(raw_imu_topic, "/roadstar/drivers/novatel/raw_imu",
              "raw_imu topic name");
DEFINE_string(ins_stat_topic, "/roadstar/drivers/novatel/ins_stat",
              "ins stat topic name");
DEFINE_string(ins_status_topic, "/roadstar/drivers/novatel/ins_status",
              "ins status topic name");
DEFINE_string(gnss_status_topic, "/roadstar/drivers/novatel/gnss_status",
              "gnss status topic name");
DEFINE_string(gnss_raw_data_topic, "/roadstar/drivers/novatel/raw_data",
              "gnss raw data topic name");
DEFINE_string(gnss_best_pose_topic, "/roadstar/drivers/novatel/best_pose",
              "gnss best pose topic name");
DEFINE_string(stream_status_topic, "/roadstar/drivers/novatel/stream_status",
              "stream status topic name");
DEFINE_string(rtcm_data_topic, "/roadstar/drivers/novatel/rtcm_data",
              "rtcm data topic name");
DEFINE_string(pluto_imu_topic, "/roadstar/drivers/imu/pluto",
              "pluto imu data topic name");
DEFINE_string(asensing_ins_topic, "/roadstar/drivers/asensing/ins",
              "asensing ins data topic name");
DEFINE_string(asensing_ins_status_topic, "/roadstar/drivers/asensing/ins_stat",
              "asensing ins status data topic name");
DEFINE_string(gnss_ins_topic, "/roadstar/drivers/gnss/ins",
              "unified gnss ins data topic name");
// perception internal msgs
DEFINE_string(lidar_perception_topic, "/roadstar/perception/lidar_perception",
              "lidar perception topic name");
DEFINE_string(old_camera_obstacles_topic,
              "/roadstar/perception/old_camera_obstacles",
              "old camera obstacles topic name");
DEFINE_string(camera_obstacles_topic, "/roadstar/perception/camera_obstacles",
              "camera obstacles topic name");
DEFINE_string(lidar_obstacles_topic, "/roadstar/perception/lidar_obstacles",
              "lidar obstacles topic name");
DEFINE_string(radar_obstacles_topic, "/roadstar/perception/radar_obstacles",
              "radar obstacles topic name");
// integration test internal msgs
DEFINE_string(velo64_packets_topic, "/velodyne_packets",
              "velo64 packets topic name");
DEFINE_string(hesai_packets_topic, "/roadstar/drivers/Pandar40/Packets",
              "velo64 packets topic name");
DEFINE_string(vlp1_packets_topic, "/velodyne_packets_1",
              "vlp 1 packets topic name");
DEFINE_string(vlp2_packets_topic, "/velodyne_packets_2",
              "vlp 2 packets topic name");
DEFINE_string(vlp3_packets_topic, "/velodyne_packets_3",
              "vlp 3 packets topic name");
DEFINE_string(sync_lidar_head_topic, "/roadstar/drivers/sync/head",
              "sync head topic name");
DEFINE_string(sync_lidar_front_left_topic, "/roadstar/drivers/sync/front_left",
              "sync front left topic name");
DEFINE_string(sync_lidar_front_right_topic,
              "/roadstar/drivers/sync/front_right",
              "sync front right topic name");
DEFINE_string(sync_lidar_mid_left_topic, "/roadstar/drivers/sync/mid_left",
              "sync mid left topic name");
DEFINE_string(sync_lidar_mid_right_topic, "/roadstar/drivers/sync/mid_right",
              "sync mid right topic name");
DEFINE_string(sync_lidar_tail_left_topic, "/roadstar/drivers/sync/tail_left",
              "sync tail left topic name");
DEFINE_string(sync_lidar_tail_right_topic, "/roadstar/drivers/sync/tail_right",
              "sync tail right topic name");

DEFINE_string(message_service_status_topic, "/roadstar/message_service_status",
              "Message service status topic name");
