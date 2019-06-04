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

#include "modules/common/roadstar_app.h"

#include <csignal>
#include <string>
#include <vector>

#include "gflags/gflags.h"
#include "modules/common/log.h"
#include "modules/common/message/message_service.h"
#include "modules/common/status/status.h"
#include "modules/common/util/string_util.h"

#include "ros/include/ros/ros.h"

namespace roadstar {
namespace common {

void RoadstarApp::SetCallbackThreadNumber(uint32_t callback_thread_num) {
  CHECK_GE(callback_thread_num, 1);
  callback_thread_num_ = callback_thread_num;
}

void RoadstarApp::ExportFlags() const {
  const auto export_file = util::StrCat(FLAGS_log_dir, "/", Name(), ".flags");
  std::ofstream fout(export_file);
  CHECK(fout) << "Cannot open file " << export_file;

  std::vector<gflags::CommandLineFlagInfo> flags;
  gflags::GetAllFlags(&flags);
  for (const auto &flag : flags) {
    fout << "# " << flag.type << ", default=" << flag.default_value << "\n"
         << "# " << flag.description << "\n"
         << "--" << flag.name << "=" << flag.current_value << "\n"
         << std::endl;
  }
}

int RoadstarApp::Spin() {
  auto status = Init();
  if (!status.ok()) {
    AERROR << Name() << " Init failed: " << status;
    return -1;
  }

  ros::NodeHandle nh;
  std::signal(SIGINT, roadstar::common::RoadstarAppSigintHandler);
  ros::AsyncSpinner spinner(callback_thread_num_);
  status = Start();
  if (!status.ok()) {
    AERROR << Name() << " Start failed: " << status;
    return -2;
  }
  spinner.start();
  ros::waitForShutdown();

  Stop();
  AINFO << Name() << " exited.";
  return 0;
}

void RoadstarAppSigintHandler(int signal_num) {
  AINFO << "Received signal: " << signal_num;
  if (signal_num != SIGINT) {
    return;
  }
  bool static is_stopping = false;
  if (is_stopping) {
    return;
  }
  is_stopping = true;
  ros::shutdown();
  message::MessageService::instance()->ShutDown();
}

}  // namespace common
}  // namespace roadstar
