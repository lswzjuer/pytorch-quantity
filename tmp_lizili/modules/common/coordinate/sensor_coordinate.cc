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

#include "modules/common/coordinate/sensor_coordinate.h"

#include <cmath>

#include "Eigen/Eigen"
#include "gflags/gflags.h"
#include "yaml-cpp/yaml.h"

#include "modules/common/adapters/adapter_manager.h"
#include "modules/common/coordinate/proto/transform.pb.h"
#include "modules/common/log.h"
#include "modules/common/util/file.h"

namespace roadstar {
namespace common {

using roadstar::common::adapter::AdapterManager;
using roadstar::common::util::GetAbsolutePath;
using roadstar::common::util::GetProtoFromFile;

SensorCoordinate::SensorCoordinate() {
  std::string config_path =
      GetAbsolutePath(FLAGS_calibration_config_path, "calibration_conf.config");
  if (!GetProtoFromFile(config_path, &config_)) {
    AFATAL << "Cannot get config proto from file " << config_path;
  }
  if (!Init()) {
    AFATAL << "SensorCoorinate init failed!";
  }
}

bool SensorCoordinate::Init() {
  std::lock_guard<std::mutex> lock(mutex_);
  return InitInternal();
}

SensorCoordinate::~SensorCoordinate() {}

bool SensorCoordinate::InitInternal() {
  if (inited_) {
    return true;
  }
  register_.clear();
  register_[common::sensor::World] = true;
  const calibration_config::CalibrationConfigs::SensorFileConfig
      &sensor_file_config = config_.sensor_file_config();
  for (auto &file_path : sensor_file_config.sensor_files()) {
    std::string file_name =
        GetAbsolutePath(FLAGS_calibration_config_path, file_path.file_name());
    if (!LoadExtrinsic(file_name, file_path.sensor_name())) {
      return false;
    }
    register_[file_path.sensor_name()] = true;
  }
  inited_ = true;
  return true;
}

bool SensorCoordinate::IsRegistered(const sensor::SensorSource &frame) {
  if (!register_[frame]) {
    AFATAL << "No such frame : " << roadstar::common::sensor::Name(frame)
           << " Please check!";
  }
  return true;
}

CoordTransD SensorCoordinate::GetCoordTrans(
    const sensor::SensorSource &frame,
    const sensor::SensorSource &child_frame) {
  instance()->IsRegistered(frame);
  instance()->IsRegistered(child_frame);

  if (frame == common::sensor::World || child_frame == common::sensor::World) {
    AFATAL << "Unable to get CoordTrans without localization";
  }

  return instance()->coord_trans_map_[frame].Inv() *
         instance()->coord_trans_map_[child_frame];
}

CoordTransD SensorCoordinate::GetCoordTrans(
    const sensor::SensorSource &frame, const sensor::SensorSource &child_frame,
    const Localization &loc) {
  instance()->IsRegistered(frame);
  instance()->IsRegistered(child_frame);

  if (frame == common::sensor::World || child_frame == common::sensor::World) {
    Eigen::Vector3d tran;
    tran(0) = loc.utm_x();
    tran(1) = loc.utm_y();
    tran(2) = loc.utm_z();
    CoordTransD coord_trans(loc.heading(), tran);
    if (child_frame == common::sensor::World) {
      return instance()->coord_trans_map_[frame].Inv() * coord_trans.Inv();
    } else {
      return coord_trans * instance()->coord_trans_map_[child_frame];
    }
  } else {
    return GetCoordTrans(frame, child_frame);
  }
}

CoordTransD SensorCoordinate::GetCoordTrans(
    const sensor::SensorSource &sensor_name) {
  return instance()->coord_trans_map_[sensor_name];
}

bool SensorCoordinate::LoadExtrinsic(const std::string &file_name,
                                     const sensor::SensorSource &sensor_name) {
  calibration::Transform transform;
  if (!GetProtoFromFile(file_name, &transform)) {
    AFATAL << "Unable to get calibration file: " << file_name;
    return false;
  } else {
    AINFO << "\n"
          << sensor::Name(sensor_name)
          << " calibration file is: " << transform.DebugString();
  }
  Eigen::Vector3d rot, tran;
  rot(0) = transform.roll();
  rot(1) = transform.pitch();
  rot(2) = transform.heading();
  tran(0) = transform.x();
  tran(1) = transform.y();
  tran(2) = transform.z();

  CoordTransD coord_trans(rot, tran);

  coord_trans_map_[sensor_name] = coord_trans;
  return true;
}

}  // namespace common
}  // namespace roadstar
