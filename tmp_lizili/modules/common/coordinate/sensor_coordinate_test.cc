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

#include "modules/common/coordinate/sensor_coordinate.h"
#include "modules/common/adapters/adapter_manager.h"
#include "modules/common/coordinate/proto/transform.pb.h"
#include "modules/common/util/file.h"
#include "modules/msgs/localization/proto/localization.pb.h"
#include "velodyne_msgs/PointCloud.h"

#include "gtest/gtest.h"
using roadstar::common::adapter::AdapterManager;
using roadstar::common::util::GetProtoFromFile;

namespace roadstar {
namespace common {

TEST(SensorCoordinateTest, Simple) {
  FLAGS_calibration_config_path = "resources/calibration/data/test";
  auto coord_trans =
      SensorCoordinate::GetCoordTrans(sensor::VehicleCenter, sensor::LidarMain);
  calibration::Transform velo64_trans;
  auto velo64_file =
      FLAGS_calibration_config_path + "/all2imu/main_lidar_calibration.conf";
  if (!GetProtoFromFile(velo64_file, &velo64_trans)) {
    AFATAL << "Unable to get calibration file: " << velo64_file;
  }
  calibration::Transform velo16_trans;
  auto velo16_file =
      FLAGS_calibration_config_path + "/all2imu/vlp_calibration0.conf";
  if (!GetProtoFromFile(velo16_file, &velo16_trans)) {
    AFATAL << "Unable to get calibration file: " << velo16_file;
  }

  auto velo64_translation = coord_trans.GetTranslationVec3d();
  EXPECT_EQ(velo64_translation[0], velo64_trans.x());
  EXPECT_EQ(velo64_translation[1], velo64_trans.y());
  EXPECT_EQ(velo64_translation[2], velo64_trans.z());

  Eigen::Matrix4d velo64_trans_mat, velo16_trans_mat;

  Eigen::Quaterniond velo64_q =
      Eigen::AngleAxisd(velo64_trans.heading(), Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(velo64_trans.pitch(), Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(velo64_trans.roll(), Eigen::Vector3d::UnitX());
  velo64_trans_mat.setIdentity();
  velo64_trans_mat.topLeftCorner(3, 3) = velo64_q.toRotationMatrix();
  velo64_trans_mat(0, 3) = velo64_trans.x();
  velo64_trans_mat(1, 3) = velo64_trans.y();
  velo64_trans_mat(2, 3) = velo64_trans.z();

  Eigen::Quaterniond velo16_q =
      Eigen::AngleAxisd(velo16_trans.heading(), Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(velo16_trans.pitch(), Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(velo16_trans.roll(), Eigen::Vector3d::UnitX());
  velo16_trans_mat.setIdentity();
  velo16_trans_mat.topLeftCorner(3, 3) = velo16_q.toRotationMatrix();
  velo16_trans_mat(0, 3) = velo16_trans.x();
  velo16_trans_mat(1, 3) = velo16_trans.y();
  velo16_trans_mat(2, 3) = velo16_trans.z();

  coord_trans =
      SensorCoordinate::GetCoordTrans(sensor::LidarMain, sensor::LidarTailLeft);
  Eigen::Matrix4d velo16_2_velo64 =
      velo64_trans_mat.inverse() * velo16_trans_mat;
  Eigen::Matrix4d _velo16_2_velo64 = coord_trans.GetTransMat4d();
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      EXPECT_NEAR(velo16_2_velo64(i, j), _velo16_2_velo64(i, j), 1e-9);
}

}  // namespace common
}  // namespace roadstar
