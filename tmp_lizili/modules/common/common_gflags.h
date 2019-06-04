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

#ifndef MODULES_COMMON_COMMON_GFLAGS_H_
#define MODULES_COMMON_COMMON_GFLAGS_H_

#include "gflags/gflags.h"

DECLARE_bool(onboard);
DECLARE_bool(use_ros_time);
DECLARE_string(calibration_config_path);
DECLARE_string(vehicle_name);
DECLARE_string(vehicle_config_path);
DECLARE_string(hdmap_rpc_service_address);
DECLARE_string(adapter_config_path);

#endif /* MODULES_COMMON_COMMON_GFLAGS_H_ */
