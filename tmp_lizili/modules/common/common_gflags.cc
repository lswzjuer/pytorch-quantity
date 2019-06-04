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

#include "modules/common/common_gflags.h"

DEFINE_bool(onboard, true, "Onboard or not.");
DEFINE_bool(use_ros_time, false,
            "Whether Clock::Now() gets time from system_clock::now() or from "
            "ros::Time::now().");
DEFINE_string(calibration_config_path,
              "resources/calibration/data/shiyan_truck2",
              "calibration config file direcotry.");
DEFINE_string(vehicle_name, "shiyan_truck1", "vehicle name");
DEFINE_string(vehicle_config_path, "config/vehicle_data/vehicle_in_use.pb.txt",
              "the file path of vehicle config file");
DEFINE_string(hdmap_rpc_service_address, "0.0.0.0:9999",
              "tutorials rpc service address.");
DEFINE_string(adapter_config_path, "", "the file path of adapter config file");
