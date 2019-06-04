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

#ifndef MODULES_COMMON_COORDINATE_CAMERA_H_
#define MODULES_COMMON_COORDINATE_CAMERA_H_

#include <Eigen/Core>
#include <Eigen/Dense>

#include <algorithm>
#include <string>
#include <vector>

namespace roadstar {
namespace common {

template <typename T>
struct CameraParams {
  Eigen::Matrix<T, 3, 3> intrinsic;
  Eigen::Matrix<T, 3, 1> tvec;
  Eigen::Matrix<T, 3, 1> rvec;
  Eigen::Matrix<T, 5, 1> dist_coeff;
  T image_height;
  T image_width;
};

/**@brief camera intrinsic of pin-hole camera model*/
template <typename T>
class CameraModel {
 public:
  CameraModel() {
    intrinsic_.setIdentity();
    Set(intrinsic_, 1, 1);
  }

  void Set(const Eigen::Matrix<T, 3, 3> &intrinsic, T w, T h) {
    intrinsic_ = intrinsic;
    focal_length_.x() = intrinsic_(0, 0);
    focal_length_.y() = intrinsic_(1, 1);
    center_.x() = intrinsic_(0, 2);
    center_.y() = intrinsic_(1, 2);
    width_ = w;
    height_ = h;
  }

  void Set(T focal_length_x, T focal_length_y, T center_x, T center_y, T w,
           T h) {
    focal_length_.x() = focal_length_x;
    focal_length_.y() = focal_length_y;
    center_.x() = center_x;
    center_.y() = center_y;
    width_ = w;
    height_ = h;
    intrinsic_(0, 0) = focal_length_.x();
    intrinsic_(1, 1) = focal_length_.y();
    intrinsic_(0, 2) = center_.x();
    intrinsic_(1, 2) = center_.y();
  }

  void Set(const Eigen::Matrix<T, 3, 1> &rvec,
           const Eigen::Matrix<T, 3, 1> &tvec) {
    rvec_ = rvec;
    tvec_ = tvec;
    Rodrigues(rvec_, &extrinsic_);
    extrinsic_inv_ = extrinsic_.inverse();
  }

  void Set(const Eigen::Matrix<T, 3, 3> &rmat,
           const Eigen::Matrix<T, 3, 1> &tvec) {
    // TODO(qianwei): since only has one-way Rodrigues transformation
    rvec_ << 0., 0., 0.;
    tvec_ = tvec;
    extrinsic_ = rmat;
    extrinsic_inv_ = extrinsic_.inverse();
  }

  void Set(const T &height) {
    lidar_height_ = height;
  }

  virtual Eigen::Matrix<T, 3, 1> Project3d(
      const Eigen::Matrix<T, 3, 1> &pt3d) const {
    return extrinsic_ * pt3d + tvec_;
  }

  /**@brief Project a 3D point on an image. */
  virtual Eigen::Matrix<T, 2, 1> Project(
      const Eigen::Matrix<T, 3, 1> &pt3d) const {
    Eigen::Matrix<T, 3, 1> local_pt3d = extrinsic_ * pt3d + tvec_;
    Eigen::Matrix<T, 2, 1> pt2d;
    pt2d[0] = local_pt3d[0] / local_pt3d[2];
    pt2d[1] = local_pt3d[1] / local_pt3d[2];

    return PixelDenormalize(pt2d);
  }

  /**@brief Project a 3D point on an image. */
  virtual Eigen::Matrix<T, 2, 1> Project(
      const Eigen::Matrix<T, 3, 1> &pt3d,
      const Eigen::Matrix<T, 3, 3> &extrinsic,
      const Eigen::Matrix<T, 3, 1> &tvec) const {
    Eigen::Matrix<T, 3, 1> local_pt3d = extrinsic * pt3d + tvec;
    Eigen::Matrix<T, 2, 1> pt2d;
    pt2d[0] = local_pt3d[0] / local_pt3d[2];
    pt2d[1] = local_pt3d[1] / local_pt3d[2];

    return PixelDenormalize(pt2d);
  }

  /**@brief Project a 3D point on an image. */
  virtual std::vector<Eigen::Matrix<T, 2, 1>> Project(
      const std::vector<Eigen::Matrix<T, 3, 1>> &pt3ds,
      const Eigen::Matrix<T, 3, 1> &rvec,
      const Eigen::Matrix<T, 3, 1> &tvec) const {
    Eigen::Matrix<T, 3, 3> extrinsic;
    Rodrigues(rvec, &extrinsic);
    std::vector<Eigen::Matrix<T, 2, 1>> pt2ds;
    pt2ds.reserve(pt3ds.size());
    for (auto pt3d : pt3ds) {
      pt2ds.emplace_back(Project(pt3d, extrinsic, tvec));
    }
    return pt2ds;
  }

  /**@brief Convert rvec to rotation matrix */
  void Rodrigues(const Eigen::Matrix<T, 3, 1> &rvec,
                 Eigen::Matrix<T, 3, 3> *rot) const {
    T theta =
        std::sqrt(rvec(0) * rvec(0) + rvec(1) * rvec(1) + rvec(2) * rvec(2));
    if (theta < 1e-9) {
      rot->setIdentity();
    } else {
      T c = std::cos(theta);
      T s = std::sin(theta);
      T c1 = 1. - c;
      T itheta = theta ? 1. / theta : 0.;
      Eigen::Matrix<T, 3, 1> r;
      r(0) = rvec(0) * itheta;
      r(1) = rvec(1) * itheta;
      r(2) = rvec(2) * itheta;
      Eigen::Matrix<T, 3, 3> rrt, r_x;
      rrt << r(0) * r(0), r(0) * r(1), r(0) * r(2), r(1) * r(0), r(1) * r(1),
          r(1) * r(2), r(2) * r(0), r(2) * r(1), r(2) * r(2);
      r_x << 0, -r(2), r(1), r(2), 0, -r(0), -r(1), r(0), 0;
      *rot = c * Eigen::Matrix<T, 3, 3>::Identity() + c1 * rrt + s * r_x;
    }
  }

  /**@brief Unproject a pixel to 3D point on a given XY plane, where z = 1 */
  virtual Eigen::Matrix<T, 3, 1> Unproject(
      const Eigen::Matrix<T, 2, 1> &pt2d) const {
    Eigen::Matrix<T, 3, 1> pt3d;
    Eigen::Matrix<T, 2, 1> pt2d_tmp = PixelNormalize(pt2d);
    pt3d[2] =
        (-lidar_height_ + extrinsic_inv_(2, 0) * tvec_(0) +
         extrinsic_inv_(2, 1) * tvec_(1) + extrinsic_inv_(2, 2) * tvec_(2)) /
        (extrinsic_inv_(2, 0) * pt2d_tmp[0] +
         extrinsic_inv_(2, 1) * pt2d_tmp[1] + extrinsic_inv_(2, 2));
    pt3d[0] = pt2d_tmp[0] * pt3d[2];
    pt3d[1] = pt2d_tmp[1] * pt3d[2];
    if (pt3d[2] > 0) {
      pt3d = extrinsic_inv_ * (pt3d - tvec_);
      pt3d[2] = 1;
    } else {
      pt3d = extrinsic_inv_ * (pt3d - tvec_);
      pt3d[2] = -1;
    }
    return pt3d;
  }

  /**@brief Get the focal length. */
  inline Eigen::Matrix<T, 2, 1> focal_length() const {
    return focal_length_;
  }
  /**@brief Get the optical center. */
  inline Eigen::Matrix<T, 2, 1> center() const {
    return center_;
  }

  /**@brief Get the intrinsic matrix. */
  inline Eigen::Matrix<T, 3, 3> intrinsic() {
    return intrinsic_;
  }

  /**@brief Get the extrinsic matrix. */
  inline Eigen::Matrix<T, 3, 3> extrinsic() {
    return extrinsic_;
  }

  /**@brief Get the tvec. */
  inline Eigen::Matrix<T, 3, 1> tvec() {
    return tvec_;
  }

  /**@brief Get the rvec. */
  inline Eigen::Matrix<T, 3, 1> rvec() {
    return rvec_;
  }

  /**@brief Get the lidar height. */
  inline T lidar_height() {
    return lidar_height_;
  }

  /**@brief Get the image width */
  inline T width() const {
    return width_;
  }
  /**@brief Get the image height */
  inline T height() const {
    return height_;
  }

  const virtual std::string DebugString() const {
    std::ostringstream oss;
    oss << "camera intrinsic: \n"
        << intrinsic_ << "\n camera width " << width_ << "\n camera height "
        << height_;
    return oss.str();
  }

 protected:
  /**@brief Normalize a 2D pixel. Convert a 2D pixel as if the image is taken
   * with a camera,
   * whose K = identity matrix. */
  virtual Eigen::Matrix<T, 2, 1> PixelNormalize(
      const Eigen::Matrix<T, 2, 1> &pt2d) const {
    Eigen::Matrix<T, 2, 1> p;
    p[0] = (pt2d[0] - center_.x()) / focal_length_.x();
    p[1] = (pt2d[1] - center_.y()) / focal_length_.y();

    return p;
  }

  /**@brief Denormalize a 2D pixel. Convert a 2D pixel as if the image is taken
   * with a camera,
   * whose K = intrinsic_. */
  virtual Eigen::Matrix<T, 2, 1> PixelDenormalize(
      const Eigen::Matrix<T, 2, 1> &pt2d) const {
    Eigen::Matrix<T, 2, 1> p;
    p[0] = pt2d[0] * focal_length_.x() + center_.x();
    p[1] = pt2d[1] * focal_length_.y() + center_.y();

    return p;
  }

 protected:
  /**@brief The camera intrinsic matrix. */
  Eigen::Matrix<T, 3, 3> intrinsic_;
  /**@brief The camera extrinsic matrix. */
  Eigen::Matrix<T, 3, 3> extrinsic_;
  /**@brief The camera extrinsic inv matrix. */
  Eigen::Matrix<T, 3, 3> extrinsic_inv_;
  /**@brief The camera translation vector. */
  Eigen::Matrix<T, 3, 1> tvec_;
  /**@brief The camera rotation vector. */
  Eigen::Matrix<T, 3, 1> rvec_;
  /**@brief The focal length. */
  Eigen::Matrix<T, 2, 1> focal_length_;
  /**@brief The optical center.  */
  Eigen::Matrix<T, 2, 1> center_;
  /**@brief Image width */
  T width_;
  /**@brief Image height */
  T height_;
  /**@brief Lidar height */
  T lidar_height_;
};

/**@brief camera intrinsic of pin-hole camera model with distortion*/
template <typename T>
class CameraDistortModel : public CameraModel<T> {
 public:
  /**@brief The default constructor. */
  CameraDistortModel() {
    distort_params_.setZero();
  }

  /**@brief Set the distortion parameters to be zeros */
  void SetZeroDistortParams() {
    distort_params_.setZero();
  }

  /**@brief Set the distortion parameters. */
  void SetDistortParams(T d0, T d1, T d2, T d3, T d4) {
    distort_params_[0] = d0;
    distort_params_[1] = d1;
    distort_params_[2] = d2;
    distort_params_[3] = d3;
    distort_params_[4] = d4;
  }

  /**@brief Set the distortion parameters. */
  inline void SetDistortParams(const Eigen::Matrix<T, 5, 1> &params) {
    distort_params_ = params;
  }

  /**@brief Set the distortion map. */
  void SetDistortMaps(const cv::Mat &map1, const cv::Mat &map2) {
    map1_ = map1;
    map2_ = map2;
  }

  /**@brief Get the distortion parameters. */
  inline Eigen::Matrix<T, 5, 1> distort_params() {
    return distort_params_;
  }

  const std::string DebugString() const {
    std::ostringstream oss;
    oss << CameraModel<T>::DebugString() << "\ncamera distort param: \n"
        << distort_params_;
    return oss.str();
  }

  void Undistort(const cv::Mat &src, cv::Mat *dst) {
    cv::remap(src, *dst, map1_, map2_, cv::INTER_LINEAR);
  }

 protected:
  /**@brief Normalize a 2D pixel. Convert a 2D pixel as if the image is taken
   * with a camera,
   * whose K = identity matrix. */
  Eigen::Matrix<T, 2, 1> PixelNormalize(
      const Eigen::Matrix<T, 2, 1> &pt2d) const override {
    Eigen::Matrix<T, 2, 1> pt2d_distort = CameraModel<T>::PixelNormalize(pt2d);
    Eigen::Matrix<T, 2, 1> pt2d_undistort = pt2d_distort;  // Initial guess
    for (unsigned int i = 0; i < 20; ++i) {
      T r_sq = pt2d_undistort[0] * pt2d_undistort[0] +
               pt2d_undistort[1] * pt2d_undistort[1];
      T k_radial = 1.0 + distort_params_[0] * r_sq +
                   distort_params_[1] * r_sq * r_sq +
                   distort_params_[4] * r_sq * r_sq * r_sq;
      T delta_x_0 =
          2 * distort_params_[2] * pt2d_undistort[0] * pt2d_undistort[1] +
          distort_params_[3] *
              (r_sq + 2 * pt2d_undistort[0] * pt2d_undistort[0]);
      T delta_x_1 =
          distort_params_[2] *
              (r_sq + 2 * pt2d_undistort[1] * pt2d_undistort[1]) +
          2 * distort_params_[3] * pt2d_undistort[0] * pt2d_undistort[1];
      pt2d_undistort[0] = (pt2d_distort[0] - delta_x_0) / k_radial;
      pt2d_undistort[1] = (pt2d_distort[1] - delta_x_1) / k_radial;
    }
    return pt2d_undistort;
  }

  /**@brief Denormalize a 2D pixel. Convert a 2D pixel as if the image is taken
   * with a camera,
   * whose K = intrinsic_. */
  Eigen::Matrix<T, 2, 1> PixelDenormalize(
      const Eigen::Matrix<T, 2, 1> &pt2d) const override {
    // Add distortion
    T r_sq = pt2d[0] * pt2d[0] + pt2d[1] * pt2d[1];
    Eigen::Matrix<T, 2, 1> pt2d_radial =
        pt2d *
        (1 + distort_params_[0] * r_sq + distort_params_[1] * r_sq * r_sq +
         distort_params_[4] * r_sq * r_sq * r_sq);
    Eigen::Matrix<T, 2, 1> dpt2d;
    dpt2d[0] = 2 * distort_params_[2] * pt2d[0] * pt2d[1] +
               distort_params_[3] * (r_sq + 2 * pt2d[0] * pt2d[0]);
    dpt2d[1] = distort_params_[2] * (r_sq + 2 * pt2d[1] * pt2d[1]) +
               2 * distort_params_[3] * pt2d[0] * pt2d[1];

    Eigen::Matrix<T, 2, 1> pt2d_distort;
    pt2d_distort[0] = pt2d_radial[0] + dpt2d[0];
    pt2d_distort[1] = pt2d_radial[1] + dpt2d[1];
    // Add intrinsic K
    return CameraModel<T>::PixelDenormalize(pt2d_distort);
  }

 protected:
  /**@brief The distortion parameters.
   *
   * See here for the definition of the parameters:
   * http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/parameters.html
   */
  Eigen::Matrix<T, 5, 1> distort_params_;
  /**@brief Undistort map param */
  cv::Mat map1_;
  cv::Mat map2_;
};

typedef CameraModel<double> CameraModelD;
typedef CameraDistortModel<double> CameraDistortModelD;

}  // namespace common
}  // namespace roadstar

#endif  // MODULES_COMMON_COORDINATE_CAMERA_H_
