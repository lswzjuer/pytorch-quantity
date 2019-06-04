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

#include "modules/common/coordinate/camera_coordinate.h"
#include <cmath>
#include "Eigen/Eigen"
#include "gflags/gflags.h"
#include "modules/common/log.h"
#include "modules/common/sensor_source.h"
#include "yaml-cpp/yaml.h"

namespace roadstar {
namespace common {

using roadstar::common::util::GetAbsolutePath;
using roadstar::common::util::GetProtoFromFile;

bool CameraCoordinate::Init() {
  std::lock_guard<std::mutex> lock(mutex_);
  return InitInternal();
}

CameraCoordinate::CameraCoordinate() {
  std::string config_path =
      GetAbsolutePath(FLAGS_calibration_config_path, "calibration_conf.config");
  if (!GetProtoFromFile(config_path, &config_)) {
    AFATAL << "Cannot get config proto from file " << config_path;
  }
  if (!Init()) {
    AFATAL << "CameraCoordinate init failed!";
  }
}

CameraCoordinate::~CameraCoordinate() {}

bool CameraCoordinate::InitInternal() {
  if (inited_) {
    return true;
  }

  for (auto &camera_file : config_.sensor_file_config().camera_files()) {
    auto camera_name = camera_file.sensor_name();
    auto file_name =
        GetAbsolutePath(FLAGS_calibration_config_path, camera_file.file_name());
    if (!InitCameraModel(file_name, camera_name)) {
      return false;
    }
  }

  AINFO << "finish to load Calibration Configs.";

  inited_ = true;
  return inited_;
}

bool CameraCoordinate::InitCameraModel(const std::string &file_path,
                                       const sensor::SensorSource &name) {
  CameraConf camera_conf;
  if (!GetProtoFromFile(file_path, &camera_conf)) {
    AERROR << "Unable to get camera calibration file: " << name;
    return false;
  } else {
    ADEBUG << "\ncamera calibration file is: \n" << camera_conf.DebugString();
  }
  double image_height = camera_conf.image_height();
  double image_width = camera_conf.image_width();
  auto &camera_matrix = camera_conf.camera_matrix();
  camera_distort_model_[name].Set(camera_matrix.fx(), camera_matrix.fy(),
                                  camera_matrix.cx(), camera_matrix.cy(),
                                  image_width, image_height);
  auto &dist_coeffs = camera_conf.dist_coeffs();
  camera_distort_model_[name].SetDistortParams(
      dist_coeffs.k1(), dist_coeffs.k2(), dist_coeffs.p1(), dist_coeffs.p2(),
      dist_coeffs.k3());
  camera_distort_model_[name].Set(camera_conf.height());

  auto &extrinsic = camera_conf.lidar64_trans();
  Eigen::Matrix<double, 3, 1> rvec, tvec;
  rvec(0) = extrinsic.rvec(0);
  rvec(1) = extrinsic.rvec(1);
  rvec(2) = extrinsic.rvec(2);
  tvec(0) = extrinsic.tvec(0);
  tvec(1) = extrinsic.tvec(1);
  tvec(2) = extrinsic.tvec(2);
  camera_distort_model_[name].Set(rvec, tvec);

  cv::Mat cam_mat = cv::Mat::eye(3, 3, CV_64F);
  cv::Mat dist_coeff_mat = cv::Mat(5, 1, CV_64F);
  cam_mat.at<double>(0, 0) = camera_matrix.fx();
  cam_mat.at<double>(0, 2) = camera_matrix.cx();
  cam_mat.at<double>(1, 1) = camera_matrix.fy();
  cam_mat.at<double>(1, 2) = camera_matrix.cy();
  dist_coeff_mat.at<double>(0, 0) = dist_coeffs.k1();
  dist_coeff_mat.at<double>(1, 0) = dist_coeffs.k2();
  dist_coeff_mat.at<double>(2, 0) = dist_coeffs.p1();
  dist_coeff_mat.at<double>(3, 0) = dist_coeffs.p2();
  dist_coeff_mat.at<double>(4, 0) = dist_coeffs.k3();
  cv::Mat map1, map2;
  cv::initUndistortRectifyMap(cam_mat, dist_coeff_mat, cv::Mat(), cam_mat,
                              cv::Size(image_width, image_height), CV_32FC1,
                              map1, map2);
  camera_distort_model_[name].SetDistortMaps(map1, map2);

  return true;
}

CameraDistortModelD CameraCoordinate::GetCameraModel(
    const sensor::SensorSource &name) {
  if (instance()->camera_distort_model_.count(name)) {
    std::lock_guard<std::mutex> lock(instance()->model_mutex_);
    return instance()->camera_distort_model_[name];
  } else {
    AFATAL << "There is no " << common::sensor::Name(name)
           << " calibration file.";
  }
}

CameraModelD CameraCoordinate::GetOriCameraModel(
    const sensor::SensorSource &name) {
  if (instance()->camera_distort_model_.count(name)) {
    std::lock_guard<std::mutex> lock(instance()->model_mutex_);
    return instance()->camera_distort_model_[name];
  } else {
    AFATAL << "There is no " << common::sensor::Name(name)
           << " calibration file.";
  }
}

void CameraCoordinate::UpdateExtrinsic(
    const sensor::SensorSource &camera_name,
    const Eigen::Matrix<double, 3, 1> &rvec,
    const Eigen::Matrix<double, 3, 1> &tvec) {
  std::lock_guard<std::mutex> lock(instance()->model_mutex_);
  instance()->camera_distort_model_[camera_name].Set(rvec, tvec);
}

}  // namespace common
}  // namespace roadstar
