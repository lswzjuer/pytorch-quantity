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

#include "modules/common/coordinate/camera_coordinate.h"
#include <opencv2/core/core.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "gtest/gtest.h"
#include "modules/common/coordinate/proto/camera_conf.pb.h"

namespace roadstar {
namespace common {

using roadstar::common::util::GetProtoFromFile;

TEST(CameraCoordinateTest, Simple) {
  calibration_config::CalibrationConfigs config;
  std::string config_path =
      FLAGS_calibration_config_path + "/calibration_conf.config";
  if (!GetProtoFromFile(config_path, &config)) {
    AERROR << "Cannot get config proto from file " << config_path;
  }
  std::string file_name = FLAGS_calibration_config_path + "/camera/tail_right";
  CameraConf camera_conf;
  if (!GetProtoFromFile(file_name, &camera_conf)) {
    AERROR << "Unable to get camera calibration file: " << file_name;
  }

  cv::Mat rvec = cv::Mat(3, 1, CV_64F);
  cv::Mat tvec = cv::Mat(3, 1, CV_64F);
  cv::Mat rmat;
  cv::Mat rmat_inv;
  cv::Mat camera_matrix = cv::Mat(3, 3, CV_64F);
  cv::Mat dist_coeffs = cv::Mat(5, 1, CV_64F);

  double rvec_rep[3], tvec_rep[3];
  std::memcpy(rvec_rep, camera_conf.lidar64_trans().rvec().data(),
              3 * sizeof(double));
  std::memcpy(tvec_rep, camera_conf.lidar64_trans().tvec().data(),
              3 * sizeof(double));
  // truck2 tail right parameters
  rvec.at<double>(0, 0) = rvec_rep[0];
  rvec.at<double>(1, 0) = rvec_rep[1];
  rvec.at<double>(2, 0) = rvec_rep[2];

  tvec.at<double>(0, 0) = tvec_rep[0];
  tvec.at<double>(1, 0) = tvec_rep[1];
  tvec.at<double>(2, 0) = tvec_rep[2];

  auto &camera_mat = camera_conf.camera_matrix();
  double camera_matrix_array[9] = {camera_mat.fx(),
                                   0,
                                   camera_mat.cx(),
                                   0,
                                   camera_mat.fy(),
                                   camera_mat.cy(),
                                   0,
                                   0,
                                   1};
  auto &dist_coeff = camera_conf.dist_coeffs();
  double dist_coeffs_array[5] = {dist_coeff.k1(), dist_coeff.k2(),
                                 dist_coeff.p1(), dist_coeff.p2(), 0};

  std::memcpy(camera_matrix.data, camera_matrix_array, 3 * 3 * sizeof(double));
  std::memcpy(dist_coeffs.data, dist_coeffs_array, 5 * 1 * sizeof(double));

  cv::Mat empty;
  cv::Rodrigues(rvec, rmat);
  rmat_inv = rmat.inv();

  std::vector<cv::Point3d> object_points;
  std::vector<cv::Point2d> image_points;
  object_points.resize(1);
  object_points[0] = cv::Point3d(10.0, 20.0, 1.0);

  cv::projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs,
                    image_points);

  cv::Point2d img_pt = image_points[0];

  // Formula for project
  // https://docs.opencv.org/2.4/
  // modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
  cv::Mat pts = cv::Mat(3, 1, CV_64F);
  pts.at<double>(0, 0) = 10.0;
  pts.at<double>(1, 0) = 20.0;
  pts.at<double>(2, 0) = 1.0;

  pts = rmat * pts + tvec;
  pts.at<double>(0, 0) = pts.at<double>(0, 0) / pts.at<double>(2, 0);
  pts.at<double>(1, 0) = pts.at<double>(1, 0) / pts.at<double>(2, 0);
  pts.at<double>(2, 0) = 1.;
  double x = pts.at<double>(0, 0);
  double y = pts.at<double>(1, 0);
  double r_sq = x * x + y * y;
  double c_dist = 1 + dist_coeffs_array[0] * r_sq +
                  dist_coeffs_array[1] * r_sq * r_sq +
                  dist_coeffs_array[4] * r_sq * r_sq * r_sq;
  double tx = x * c_dist + 2 * dist_coeffs_array[2] * x * y +
              dist_coeffs_array[3] * (r_sq + 2 * x * x);
  double ty = y * c_dist + dist_coeffs_array[2] * (r_sq + 2 * y * y) +
              2 * dist_coeffs_array[3] * x * y;
  pts.at<double>(0, 0) = tx;
  pts.at<double>(1, 0) = ty;
  pts = camera_matrix * pts;

  EXPECT_LT(fabs(pts.at<double>(0, 0) - img_pt.x), 1e-9);
  EXPECT_LT(fabs(pts.at<double>(1, 0) - img_pt.y), 1e-9);

  auto camera_name = sensor::CameraTailRight;
  CameraDistortModelD camera_model =
      CameraCoordinate::instance()->GetCameraModel(camera_name);

  Eigen::Matrix<double, 3, 1> pt3d;
  Eigen::Matrix<double, 2, 1> pt2d;
  pt3d << 10.0, 20.0, 1.0;
  pt2d = camera_model.Project(pt3d);
  EXPECT_LT(fabs(img_pt.x - pt2d[0]), 1e-9);
  EXPECT_LT(fabs(img_pt.y - pt2d[1]), 1e-9);

  cv::Mat cam_p(3, 1, CV_64F);
  double xp =
      (10 - camera_matrix.at<double>(0, 2)) / camera_matrix.at<double>(0, 0);
  double yp =
      (10 - camera_matrix.at<double>(1, 2)) / camera_matrix.at<double>(1, 1);
  cam_p.at<double>(2) =
      (-camera_conf.height() + rmat_inv.at<double>(2, 0) * tvec.at<double>(0) +
       rmat_inv.at<double>(2, 1) * tvec.at<double>(1) +
       rmat_inv.at<double>(2, 2) * tvec.at<double>(2)) /
      (rmat_inv.at<double>(2, 0) * xp + rmat_inv.at<double>(2, 1) * yp +
       rmat_inv.at<double>(2, 2));
  cam_p.at<double>(0) = xp * cam_p.at<double>(2);
  cam_p.at<double>(1) = yp * cam_p.at<double>(2);
  cv::Mat lidar_point = rmat_inv * (cam_p - tvec);

  pt2d = Eigen::Matrix<double, 2, 1>(10.0, 10.0);
  camera_model.SetZeroDistortParams();
  pt3d = camera_model.Unproject(pt2d);
  EXPECT_LT(fabs(lidar_point.at<double>(0, 0) - pt3d[0]), 1e-9);
  EXPECT_LT(fabs(lidar_point.at<double>(1, 0) - pt3d[1]), 1e-9);
}

}  // namespace common
}  // namespace roadstar
