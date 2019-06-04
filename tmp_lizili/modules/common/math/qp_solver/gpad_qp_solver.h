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
 * @file gpad.h
 * @brief Convert mpc problem to qp based problem and solve.
 */

#ifndef MODULES_COMMON_MATH_QP_SOLVER_GPAD_QP_SOLVER_H_
#define MODULES_COMMON_MATH_QP_SOLVER_GPAD_QP_SOLVER_H_

#include <vector>

#include "Eigen/Core"
#include "modules/common/math/qp_solver/qp_solver.h"

/**
 * @namespace roadstar::common::math
 * @brief roadstar::common::math
 */

namespace roadstar {
namespace common {
namespace math {

class GPADQpSolver : public QpSolver {
 public:
  GPADQpSolver(const Eigen::MatrixXd& kernel_matrix,
               const Eigen::MatrixXd& offset,
               const Eigen::MatrixXd& affine_inequality_matrix,
               const Eigen::MatrixXd& affine_inequality_boundary,
               const Eigen::MatrixXd& affine_equality_matrix,
               const Eigen::MatrixXd& affine_equality_boundary,
               const int max_iter, const double eps);
  virtual ~GPADQpSolver() = default;

  bool Solve() override;
  void set_max_iteration(const int max_iter);
  int max_iteration() const;
  void SetTerminationTolerance(const double tolerance) override {
    termination_tolerance_ = tolerance;
  }

 private:
  bool sanity_check() override;

 private:
  // equality constriant + inequality constraint
  int num_constraint_ = 0;
  // number of parameters
  int num_param_ = 0;
  int max_iteration_ = 1000;
  double termination_tolerance_ = 1.0e-2;
};
}  // namespace math
}  // namespace common
}  // namespace roadstar

#endif  // MODULES_COMMON_MATH_QP_SOLVER_GPAD_QP_SOLVER_H_
