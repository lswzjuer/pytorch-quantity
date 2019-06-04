/******************************************************************************
 * Copyright 2018 The Roadstar Authors. All Rights Reserved.
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
// SensorCalibration use yaml config to manage all your calibration
// configs.
//
// CODE SAMPLE:
// you can use such code to access your parameters:
//
//         #include "coordinate/camera_coordinate.h"
//
//
//         std::string camera_name = "head_left";
//         auto camera_model =
//           CameraCoordnate::instance()->GetCameraModel(camera_name);
//         Eigen::Matrix<T, 3, 1> pt3d;
//         auto pt2d = camera_model.Project(pt3d);
//
//

#ifndef MODULES_COMMON_COORDINATE_CAMERA_COORDINATE_H_
#define MODULES_COMMON_COORDINATE_CAMERA_COORDINATE_H_

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <typeinfo>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include "modules/common/common_gflags.h"
#include "modules/common/coordinate/camera.h"
#include "modules/common/coordinate/proto/calibration_config.pb.h"
#include "modules/common/coordinate/proto/camera_conf.pb.h"
#include "modules/common/coordinate/proto/transform.pb.h"
#include "modules/common/macro.h"
#include "modules/common/util/file.h"

namespace roadstar {
namespace common {

class CameraCoordinate {
 public:
  // thread-safe interface.
  bool Init();

  static CameraDistortModelD GetCameraModel(const sensor::SensorSource &name);

  static CameraModelD GetOriCameraModel(const sensor::SensorSource &name);

  static void UpdateExtrinsic(const common::sensor::SensorSource &camera_name,
                              const Eigen::Matrix<double, 3, 1> &rvec,
                              const Eigen::Matrix<double, 3, 1> &tvec);

 private:
  ~CameraCoordinate();
  bool InitInternal();
  bool InitCameraModel(const std::string &file_path,
                       const sensor::SensorSource &camera_model);

  std::map<common::sensor::SensorSource, CameraDistortModelD>
      camera_distort_model_;

  std::mutex mutex_;
  std::mutex model_mutex_;
  bool inited_ = false;

  calibration_config::CalibrationConfigs config_;

  DECLARE_SINGLETON(CameraCoordinate);
};

}  // namespace common
}  // namespace roadstar

#endif  // MODULES_COMMON_COORDINATE_CAMERA_COORDINATE_H_
