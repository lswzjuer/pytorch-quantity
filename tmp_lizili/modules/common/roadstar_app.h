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

#ifndef MODULES_ROADSTAR_APP_H_
#define MODULES_ROADSTAR_APP_H_

#include <csignal>
#include <string>

#include "gflags/gflags.h"
#include "modules/common/log.h"
#include "modules/common/status/status.h"

#include "ros/include/ros/ros.h"

/**
 * @namespace roadstar::common
 * @brief roadstar::common
 */
namespace roadstar {
namespace common {

/**
 * @class RoadstarApp
 *
 * @brief The base module class to define the interface of an Roadstar app.
 * An Roadstar app runs infinitely until being shutdown by SIGINT or ROS. Many
 * essential components in Roadstar, such as localization and control are
 * examples of Roadstar apps. The ROADSTAR_MAIN macro helps developer to setup
 * glog, gflag and ROS in one line.
 */
class RoadstarApp {
 public:
  /**
   * @brief module name. It is used to uniquely identify the app.
   */
  virtual std::string Name() const = 0;

  /**
   * @brief this is the entry point of an Roadstar App. It initializes the app,
   * starts the app, and stop the app when the ros has shutdown.
   */
  virtual int Spin();

  /**
   * The default destructor.
   */
  virtual ~RoadstarApp() = default;

  /**
   * @brief set the number of threads to handle ros message callbacks.
   * The default thread number is 1
   */
  void SetCallbackThreadNumber(uint32_t callback_thread_num);

 protected:
  /**
   * @brief The module initialization function. This is the first function being
   * called when the App starts. Usually this function loads the configurations,
   * subscribe the data from sensors or other modules.
   * @return Status initialization status
   */
  virtual roadstar::common::Status Init() = 0;

  /**
   * @brief The module start function. Roadstar app usually triggered to execute
   * in two ways: 1. Triggered by upstream messages, or 2. Triggered by timer.
   * If an app is triggered by upstream messages, the Start() function usually
   * register a call back function that will be called when an upstream message
   * is received. If an app is triggered by timer, the Start() function usually
   * register a timer callback function.
   * @return Status start status
   */
  virtual roadstar::common::Status Start() = 0;

  /**
   * @brief The module stop function. This function will be called when
   * after ros::shutdown() has finished. In the default ROADSTAR_MAIN macro,
   * ros::shutdown() is called when SIGINT is received.
   */
  virtual void Stop() = 0;

  /** The callback thread number
   */
  uint32_t callback_thread_num_ = 1;

 private:
  /**
   * @brief Export flag values to <FLAGS_log_dir>/<name>.flags.
   */
  void ExportFlags() const;
};

void RoadstarAppSigintHandler(int signal_num);

}  // namespace common
}  // namespace roadstar

#define ROADSTAR_MAIN(APP)                                           \
  int main(int argc, char **argv) {                                  \
    google::ParseCommandLineFlags(&argc, &argv, true);               \
    roadstar::common::InitLogging(argv[0]);                          \
    std::signal(SIGINT, roadstar::common::RoadstarAppSigintHandler); \
    APP roadstar_app_;                                               \
    ros::init(argc, argv, roadstar_app_.Name());                     \
    roadstar_app_.Spin();                                            \
    return 0;                                                        \
  }

#endif  // MODULES_ROADSTAR_APP_H_
