/******************************************************************************
 * Copyright 2019 The Roadstar Authors. All Rights Reserved.
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

#include "modules/common/visualization/remote_visualizer/remote_visualizer_gflags.h"
#include <gflags/gflags.h>

DEFINE_string(remote_visualizer_address, "0.0.0.0:50051",
              "Address to remote visualizer server");
DEFINE_uint32(remote_visualizer_timeout, 1000,
              "Timeout for remote_visualizer client");
DEFINE_uint32(grpc_connection_maxmum_failure, 3,
              "Maximum failure time in grpc connection");
