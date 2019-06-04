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

#include "modules/common/canbus/sensor_gflags.h"

// data file
DEFINE_string(sensor_conf_file,
              "modules/common/delphi_esr/conf/mobileye_conf_dev.pb.txt",
              "Default sensor conf file");

// Canbus gflags
DEFINE_double(sensor_freq, 100,
              "Sensor feedback timer frequency -- 0 means event trigger.");

// System gflags
DEFINE_string(sensor_node_name, "", "Sensor node name.");

// driver name
DEFINE_string(canbus_driver_name, "", "Canbus driver name.");

DEFINE_string(adapter_config_filename,
              "modules/common/canbus/conf/adapter.conf",
              "The adapter config file");
