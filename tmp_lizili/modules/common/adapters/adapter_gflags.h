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

#ifndef MODULES_COMMON_ADAPTERS_ADAPTER_GFLAGS_H_
#define MODULES_COMMON_ADAPTERS_ADAPTER_GFLAGS_H_

#include "gflags/gflags.h"

DECLARE_bool(enable_adapter_dump);
DECLARE_string(camera_head_left_topic);
DECLARE_string(camera_head_right_topic);
DECLARE_string(camera_front_left_topic);
DECLARE_string(camera_front_right_topic);
DECLARE_string(camera_mid_left_topic);
DECLARE_string(camera_mid_right_topic);
DECLARE_string(camera_tail_left_topic);
DECLARE_string(camera_tail_right_topic);
DECLARE_string(camera_traffic_light_topic);
DECLARE_string(camera_head_left_proto_topic);
DECLARE_string(camera_head_right_proto_topic);
DECLARE_string(camera_front_left_proto_topic);
DECLARE_string(camera_front_right_proto_topic);
DECLARE_string(camera_mid_left_proto_topic);
DECLARE_string(camera_mid_right_proto_topic);
DECLARE_string(camera_tail_left_proto_topic);
DECLARE_string(camera_tail_right_proto_topic);
DECLARE_string(camera_traffic_light_proto_topic);
DECLARE_string(camera_obstacle_topic);
DECLARE_string(compressed_camera_head_left_topic);
DECLARE_string(compressed_camera_head_right_topic);
DECLARE_string(compressed_camera_front_left_topic);
DECLARE_string(compressed_camera_front_right_topic);
DECLARE_string(compressed_camera_mid_left_topic);
DECLARE_string(compressed_camera_mid_right_topic);
DECLARE_string(compressed_camera_tail_left_topic);
DECLARE_string(compressed_camera_tail_right_topic);
DECLARE_string(compressed_camera_traffic_light_topic);
DECLARE_string(compressed_camera_head_left_proto_topic);
DECLARE_string(compressed_camera_head_right_proto_topic);
DECLARE_string(compressed_camera_front_left_proto_topic);
DECLARE_string(compressed_camera_front_right_proto_topic);
DECLARE_string(compressed_camera_mid_left_proto_topic);
DECLARE_string(compressed_camera_mid_right_proto_topic);
DECLARE_string(compressed_camera_tail_left_proto_topic);
DECLARE_string(compressed_camera_tail_right_proto_topic);
DECLARE_string(compressed_camera_traffic_light_proto_topic);
DECLARE_string(chassis_detail_topic);
DECLARE_string(chassis_topic);
DECLARE_string(control_command_topic);
DECLARE_string(control_debug_topic);
DECLARE_string(control_status_topic);
DECLARE_string(esr_topic);
DECLARE_string(fusion_map_topic);
DECLARE_string(ins_topic);
DECLARE_string(localization_topic);
DECLARE_string(lane_detection_topic);
DECLARE_string(vision_lane_topic);
DECLARE_string(monitor_topic);
DECLARE_string(pad_topic);
DECLARE_string(planning_trajectory_topic);
DECLARE_string(radar_filter_topic);
DECLARE_string(point_cloud_topic);
DECLARE_string(hesai_point_cloud_topic);
DECLARE_string(rsds_topic);
DECLARE_string(system_status_topic);
DECLARE_string(traffic_light_detection_topic);
DECLARE_string(vlp_point_cloud1_topic);
DECLARE_string(vlp_point_cloud2_topic);
DECLARE_string(vlp_point_cloud3_topic);
DECLARE_string(vlp_point_cloud4_topic);
DECLARE_string(rslidar_mid_left_topic);
DECLARE_string(rslidar_mid_right_topic);
DECLARE_string(rslidar_top_left_topic);
DECLARE_string(rslidar_top_right_topic);
DECLARE_string(lidar_point_cloud_main_topic);
DECLARE_string(lidar_point_cloud_head_mid_topic);
DECLARE_string(lidar_point_cloud_tail_mid_topic);
DECLARE_string(lidar_point_cloud_head_left_topic);
DECLARE_string(lidar_point_cloud_head_right_topic);
DECLARE_string(lidar_point_cloud_top_left_topic);
DECLARE_string(lidar_point_cloud_top_right_topic);
DECLARE_string(lidar_point_cloud_tail_left_topic);
DECLARE_string(lidar_point_cloud_tail_right_topic);
DECLARE_string(lidar_packet_main_topic);
DECLARE_string(lidar_packet_head_mid_topic);
DECLARE_string(lidar_packet_tail_left_topic);
DECLARE_string(lidar_packet_tail_right_topic);
DECLARE_string(delphi_esr_topic);
DECLARE_string(conti_radar_topic);
DECLARE_string(conti_radar_tail_left_topic);
DECLARE_string(conti_radar_tail_right_topic);
DECLARE_string(conti_radar_head_middle_topic);
DECLARE_string(conti_radar_head_left_topic);
DECLARE_string(conti_radar_head_right_topic);
DECLARE_string(ultrasonic_radar_front_topic);
DECLARE_string(ultrasonic_radar_mid_topic);
DECLARE_string(ultrasonic_radar_rear_topic);
DECLARE_string(raw_imu_topic);
DECLARE_string(ins_stat_topic);
DECLARE_string(ins_status_topic);
DECLARE_string(gnss_status_topic);
DECLARE_string(gnss_raw_data_topic);
DECLARE_string(gnss_best_pose_topic);
DECLARE_string(stream_status_topic);
DECLARE_string(rtcm_data_topic);
DECLARE_string(pluto_imu_topic);
DECLARE_string(asensing_ins_topic);
DECLARE_string(asensing_ins_status_topic);
DECLARE_string(gnss_ins_topic);
// perception internal msgs
DECLARE_string(lidar_perception_topic);
DECLARE_string(old_camera_obstacles_topic);
DECLARE_string(camera_obstacles_topic);
DECLARE_string(lidar_obstacles_topic);
DECLARE_string(radar_obstacles_topic);
// integration test internal msgs
DECLARE_string(velo64_packets_topic);
DECLARE_string(hesai_packets_topic);
DECLARE_string(vlp1_packets_topic);
DECLARE_string(vlp2_packets_topic);
DECLARE_string(vlp3_packets_topic);
// sync
DECLARE_string(sync_lidar_head_topic);
DECLARE_string(sync_lidar_front_left_topic);
DECLARE_string(sync_lidar_front_right_topic);
DECLARE_string(sync_lidar_mid_left_topic);
DECLARE_string(sync_lidar_mid_right_topic);
DECLARE_string(sync_lidar_tail_left_topic);
DECLARE_string(sync_lidar_tail_right_topic);
// message service status
DECLARE_string(message_service_status_topic);
// Message Service flags
DECLARE_bool(enable_message_service);
DECLARE_bool(send_message_service);
#endif  // MODULES_COMMON_ADAPTERS_ADAPTER_GFLAGS_H_
