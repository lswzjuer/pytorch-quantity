/******************************************************************************
 * Copyright 2017 The Roadstar Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 * * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/
#include "modules/common/math/qp_solver/gpad_qp_solver.h"
#include <algorithm>
#include <memory>

#include "Eigen/Dense"
#include "modules/common/log.h"

namespace roadstar {
namespace common {
namespace math {

using Matrix = Eigen::MatrixXd;

/**
 * @file: gpad.cc
 **/

/* GPAD(G,W,H,F,Geq,Weq,maxiter,epsG,epsV)
  minimize 0.5*z'*H*z + F'*z
  s.t. G*z <= W
       Geq*z = Weq
*/

GPADQpSolver::GPADQpSolver(const Eigen::MatrixXd &kernel_matrix,
                           const Eigen::MatrixXd &offset,
                           const Eigen::MatrixXd &affine_inequality_matrix,
                           const Eigen::MatrixXd &affine_inequality_boundary,
                           const Eigen::MatrixXd &affine_equality_matrix,
                           const Eigen::MatrixXd &affine_equality_boundary,
                           const int max_iter, const double eps)
    : QpSolver(kernel_matrix, offset, affine_inequality_matrix,
               affine_inequality_boundary, affine_equality_matrix,
               affine_equality_boundary),
      num_constraint_(affine_equality_matrix_.rows() +
                      affine_inequality_matrix_.rows()),
      num_param_(kernel_matrix.rows()) {
  set_max_iteration(max_iter * 100);
  termination_tolerance_ = eps;
}

bool GPADQpSolver::Solve() {
  if (kernel_matrix_.rows() != kernel_matrix_.cols()) {
    AERROR << "kernel_matrix_.rows() [" << kernel_matrix_.rows()
           << "] and kernel_matrix_.cols() [" << kernel_matrix_.cols()
           << "] should be identical.";
    return false;
  }

  int keepgoing = 1;
  int i = 0;
  Matrix gapL = Eigen::MatrixXd::Zero(1, 1);
  double beta = 0;
  double L = 0;
  double epsG = termination_tolerance_;
  double epsV = termination_tolerance_;

  Matrix Matrix_constrain =
      Eigen::MatrixXd::Zero(num_constraint_, affine_inequality_matrix_.cols());

  Matrix_constrain << affine_inequality_matrix_, affine_equality_matrix_;

  Matrix Matrix_boundary = Eigen::MatrixXd::Zero(num_constraint_, 1);

  Matrix_boundary << -affine_inequality_boundary_, affine_equality_boundary_;

  Matrix y = Eigen::MatrixXd::Zero(num_constraint_, 1);

  Matrix w = Eigen::MatrixXd::Zero(num_constraint_, 1);
  Matrix y0 = Eigen::MatrixXd::Zero(num_constraint_, 1);
  Matrix s = Eigen::MatrixXd::Zero(num_constraint_, 1);
  Matrix M = Eigen::MatrixXd::Zero(num_constraint_, num_constraint_);
  Matrix iMG = Eigen::MatrixXd::Zero(num_param_, num_constraint_);
  Matrix iMc = Eigen::MatrixXd::Zero(num_param_, 1);
  Matrix z = Eigen::MatrixXd::Zero(num_param_, 1);

  M = Matrix_constrain * kernel_matrix_.inverse() *
      Matrix_constrain.transpose();

  iMG = kernel_matrix_.inverse() * Matrix_constrain.transpose();
  iMc = kernel_matrix_.inverse() * offset_;
  L = M.norm();
  L = 1 / L;

  while (keepgoing && i < max_iteration_) {
    int flag = 1;

    beta = static_cast<double>(i - 1) / (i + 50);
    beta = beta > 0 ? beta : 0;
    w = y + beta * (y - y0);
    z = -iMG * w - iMc;
    s = L * Matrix_constrain * z - L * Matrix_boundary;
    y0 = y;

    unsigned int j = s.rows();

    // check termination conditions
    while (flag && j) {
      if (s(j - 1, 0) > L * epsG) {
        flag = 0;
      } else {
        j--;
      }
    }

    if (j == 0) {
      gapL = -w.transpose() * s;

      if (gapL(0, 0) <= L * epsV) keepgoing = 0;
    }

    y = w + s;
    for (int q = 0; q <= num_constraint_ - 1; q++) {
      y(q, 0) = y(q, 0) > 0 ? y(q, 0) : 0;
    }
    i = i + 1;
  }

  if (i > max_iteration_) {
    AERROR << "GPAD solver failed due to reached max iteration";
    std::stringstream ss;
    ss << "GPAD inputs: " << std::endl;
    ss << "kernel_matrix:\n" << kernel_matrix_ << std::endl;
    ss << "offset:\n" << offset_ << std::endl;
    ss << "affine_inequality_matrix:\n"
       << affine_inequality_matrix_ << std::endl;
    ss << "affine_inequality_boundary:\n"
       << affine_inequality_boundary_ << std::endl;
    ss << "affine_equality_matrix:\n" << affine_equality_matrix_ << std::endl;
    ss << "affine_equality_boundary:\n"
       << affine_equality_boundary_ << std::endl;

    ADEBUG << ss.str();

    return false;
  }

  params_ = Eigen::MatrixXd::Zero(num_param_, 1);
  for (int c = 0; c < num_param_; ++c) {
    params_(c, 0) = z(c, 0);
  }

  return true;
}

void GPADQpSolver::set_max_iteration(const int max_iter) {
  max_iteration_ = max_iter;
}
int GPADQpSolver::max_iteration() const {
  return max_iteration_;
}

// pure virtual
bool GPADQpSolver::sanity_check() {
  return kernel_matrix_.rows() == kernel_matrix_.cols() &&
         kernel_matrix_.rows() == affine_inequality_matrix_.cols() &&
         kernel_matrix_.rows() == affine_equality_matrix_.cols() &&
         affine_equality_matrix_.rows() == affine_equality_boundary_.rows() &&
         affine_inequality_matrix_.rows() == affine_inequality_boundary_.rows();
}

}  // namespace math
}  // namespace common
}  // namespace roadstar
