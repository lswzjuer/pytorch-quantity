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

#ifndef MODULES_COMMON_COORDINATE_COORD_TRANS_H_
#define MODULES_COMMON_COORDINATE_COORD_TRANS_H_

#include <cmath>

#include "Eigen/Geometry"

namespace roadstar {
namespace common {
/* coordinate system */
//            x axis  ^
//                    |
//                    |
//                    |
//                    |
//                    |
//      y axis        |
//    <----------------
// Use standard ROS coordinate system (right-hand rule)
// The rotation order: z axis(yaw), y axis(pitch), x axis(roll)

template <typename T>
class CoordTrans {
 public:
  /**
   * @brief Constructs an identity transform
   */
  CoordTrans() {
    rot_.setZero();
    tran_.setZero();
    transform_matrix_ = GetTransMat4d(rot_, tran_);
  }

  /**
   * @brief Constructs a transform using only rot_z (around z-axis) and
   * translation(x, y, z)
   */
  CoordTrans(T yaw, Eigen::Matrix<T, 3, 1> tran) {
    rot_(0) = 0;
    rot_(1) = 0;
    rot_(2) = yaw;
    tran_ = tran;
    transform_matrix_ = GetTransMat4d(rot_, tran_);
  }

  /**
   * @brief Constructs a transform using rotation vector and translation vector
   */
  CoordTrans(Eigen::Matrix<T, 3, 1> rot, Eigen::Matrix<T, 3, 1> tran) {
    rot_ = rot;
    tran_ = tran;
    transform_matrix_ = GetTransMat4d(rot_, tran_);
  }

  /**
   * @brief Constructs a transform using rotation matrix and translation vector
   */
  CoordTrans(Eigen::Matrix<T, 3, 3> rmat, Eigen::Matrix<T, 3, 1> tran) {
    auto euler_angle = rmat.eulerAngles(2, 1, 0);
    rot_(0) = euler_angle(2);
    rot_(1) = euler_angle(1);
    rot_(2) = euler_angle(0);
    tran_ = tran;
    transform_matrix_.setIdentity();
    transform_matrix_.topLeftCorner(3, 3) = rmat;
    transform_matrix_.topRightCorner(3, 1) = tran;
  }

  /**
   * @brief Constructs a transform using transform matrix
   */
  explicit CoordTrans(Eigen::Matrix<T, 4, 4> mat) {
    Eigen::Matrix<T, 3, 3> rot_mat = mat.topLeftCorner(3, 3);
    auto euler_angle = rot_mat.eulerAngles(2, 1, 0);
    rot_(0) = euler_angle(2);
    rot_(1) = euler_angle(1);
    rot_(2) = euler_angle(0);
    tran_ = mat.topRightCorner(3, 1);
    transform_matrix_ = mat;
  }

  /**
   * @brief overload * opreator
   */
  CoordTrans operator*(const CoordTrans &t) {
    Eigen::Matrix<T, 4, 4> matrix = transform_matrix_ * t.transform_matrix_;
    return CoordTrans(matrix);
  }

  /**
   * @brief Return inverse of the transformer
   */
  CoordTrans Inv() const {
    Eigen::Matrix<T, 4, 4> matrix = transform_matrix_.inverse();
    return CoordTrans(matrix);
  }

  /**
   * @brief Getter for Transform Matrix
   * @return the 4*4 Transform Matrix
   */
  Eigen::Matrix<T, 4, 4> GetTransMat4d() const {
    return transform_matrix_;
  }

  /**
   * @brief Getter for Rotation Matrix
   * @return the 3*3 Rotation Matrix
   */
  Eigen::Matrix<T, 3, 3> GetRotMat3d() const {
    return transform_matrix_.topLeftCorner(3, 3);
  }

  /**
   * @brief Getter for Rotation Vector
   * @return the 3*1 Rotation Vector (order: x axis, y axis, z axis)
   */
  Eigen::Matrix<T, 3, 1> GetRotVec3d() const {
    return rot_;
  }

  /**
   * @brief Getter for Translation Vector
   * @return the 3*1 Translation Vector (order: x axis, y axis, z axis)
   */
  Eigen::Matrix<T, 3, 1> GetTranslationVec3d() const {
    return tran_;
  }

  /**
   * @brief Transformation for Tranform Coordinate 3d
   * @param 3*1 Coordinate before Tranfromation
   * @return 3*1 Coordinate after Tranfromation
   */
  Eigen::Matrix<T, 3, 1> TransformCoord3d(
      const Eigen::Matrix<T, 3, 1> &coordinate) const {
    return GetRotMat3d() * coordinate + tran_;
  }

  /**
   * @brief Rotation for Rotate vector 3d
   * @param 3*1 Vector before Rotation
   * @return 3*1 Vector after Rotation
   */
  Eigen::Matrix<T, 3, 1> RotateVec3d(
      const Eigen::Matrix<T, 3, 1> &coordinate) const {
    return GetRotMat3d() * coordinate;
  }

 private:
  Eigen::Matrix<T, 4, 4> GetTransMat4d(const Eigen::Matrix<T, 3, 1> rot,
                                       const Eigen::Matrix<T, 3, 1> tran) {
    Eigen::Quaternion<T> q =
        Eigen::AngleAxisd(rot(2), Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(rot(1), Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(rot(0), Eigen::Vector3d::UnitX());
    Eigen::Matrix<T, 4, 4> transform_matrix;
    transform_matrix.setIdentity();
    transform_matrix.topLeftCorner(3, 3) = q.toRotationMatrix();
    transform_matrix.topRightCorner(3, 1) = tran;
    return transform_matrix;
  }

  Eigen::Matrix<T, 3, 1> rot_;
  Eigen::Matrix<T, 3, 1> tran_;
  Eigen::Matrix<T, 4, 4> transform_matrix_;
};

typedef CoordTrans<double> CoordTransD;
typedef CoordTrans<float> CoordTransF;

}  // namespace common
}  // namespace roadstar

#endif  // MODULES_COMMON_COORDINATE_COORD_TRANS_H_
