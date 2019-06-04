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

//
// SensorCoordinate use yaml config to manage all your calibration
// configs.
//
// CODE SAMPLE:
// you can use such code to access your parameters:
//
//  #include "coordinate/sensor_coordinate.h"
//
//  /* change coordinate from VehicleCenter(imu) frame to lidar frame */
//   auto coord_trans =
//      SensorCoordinate::GetCoordinate(kMainLidar, kVechicleCenter);
//   Eigen::Vector3d vehicle_center_pt;
//   auto lidar_pt = coord_trans.TransformCoord3d(vehicle_center_pt);
//
//  /* change coordinate from lidar frame to world(utm) frame */
//  /* you have to offer localization */
//  auto loc =
//    AdapterManager::GetLocalization()->GetExpectedLocalization(timestamp);
//  auto coord_trans =
//  SensorCoordinate::GetCoordinate(kWorld, kMainLidar, loc);
//  auto utm_pt = coord_trans.TransformCoord3d(pt_lidar);
//
//  /* change velocity from lidar frame to world(utm) frame */
//  auto loc =
//    AdapterManager::GetLocalization()->GetExpectedLocalization(timestamp);
//  auto coord_trans =
//  SensorCoordinate::GetCoordinate(kWorld, kMainLidar, loc);
//  auto v_world = coord_trans.RotateVec3d(v_lidar);
//
//
//

#ifndef MODULES_COMMON_COORDINATE_SENSOR_COORDINATE_H_
#define MODULES_COMMON_COORDINATE_SENSOR_COORDINATE_H_

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"

#include "modules/common/common_gflags.h"
#include "modules/common/coordinate/coord_trans.h"
#include "modules/common/coordinate/proto/calibration_config.pb.h"
#include "modules/common/macro.h"
#include "modules/common/sensor_source.h"
#include "modules/common/util/file.h"
#include "modules/msgs/localization/proto/localization.pb.h"

namespace roadstar {
namespace common {

using roadstar::localization::Localization;

class SensorCoordinate {
 public:
  /**
   * @brief thread-safe interface
   */
  bool Init();

  /**
   * @brief Getter for Coordinate Transformer of child_frame to frame
   */
  static CoordTransD GetCoordTrans(const sensor::SensorSource &frame,
                                   const sensor::SensorSource &child_frame);

  /**
   * @brief Getter for Coordinate Transformer of child_frame to frame
   */
  static CoordTransD GetCoordTrans(const sensor::SensorSource &frame,
                                   const sensor::SensorSource &child_frame,
                                   const Localization &loc);

  /**
   * @brief Getter for Coordinate Transformer of origin calibration file
   */
  static CoordTransD GetCoordTrans(const sensor::SensorSource &sensor_name);

 private:
  ~SensorCoordinate();

  bool InitInternal();

  bool LoadExtrinsic(const std::string &file_path,
                     const sensor::SensorSource &sensor_name);

  bool IsRegistered(const sensor::SensorSource &frame);

  std::map<sensor::SensorSource, CoordTransD> coord_trans_map_;
  std::map<sensor::SensorSource, bool> register_;

  std::mutex mutex_;  // multi-thread init safe.
  bool inited_ = false;
  calibration_config::CalibrationConfigs config_;
  DECLARE_SINGLETON(SensorCoordinate);
};

}  // namespace common
}  // namespace roadstar

#endif  // MODULES_COMMON_COORDINATE_SENSOR_COORDINATE_H_
