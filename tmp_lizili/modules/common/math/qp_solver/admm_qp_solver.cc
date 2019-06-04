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
#include "modules/common/math/qp_solver/admm_qp_solver.h"
#include <Eigen/LU>
#include <algorithm>
#include <memory>
#include "Eigen/Dense"
#include "modules/common/log.h"

namespace roadstar {
namespace common {
namespace math {

using Matrix = Eigen::MatrixXd;

/**
 * @file: admm.cc
 **/

/* ADMM(G,W,H,F,Geq,Weq,maxiter,eps)
  minimize 0.5*z'*H*z + F'*z
  s.t. G*z <= W
       Geq*z = Weq
*/

ADMMQpSolver::ADMMQpSolver(const Eigen::MatrixXd &kernel_matrix,
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
  set_max_iteration(max_iter);
  termination_tolerance_ = eps;
}

bool ADMMQpSolver::Solve() {
  if (kernel_matrix_.rows() != kernel_matrix_.cols()) {
    AERROR << "kernel_matrix_.rows() [" << kernel_matrix_.rows()
           << "] and kernel_matrix_.cols() [" << kernel_matrix_.cols()
           << "] should be identical.";
    return false;
  }

  int i = 0;
  double rho = 7.0;
  double beta = 1.8;
  double resx = 0;
  double resz = 0;
  double epspri = 0;
  double epsdual = 0;
  int flag = 1;

  Matrix Matrix_constrain =
      Eigen::MatrixXd::Zero(num_constraint_, affine_inequality_matrix_.cols());

  Matrix_constrain << affine_inequality_matrix_, affine_equality_matrix_;

  Matrix Matrix_boundary = Eigen::MatrixXd::Zero(num_constraint_, 1);

  Matrix_boundary << -affine_inequality_boundary_, affine_equality_boundary_;

  Matrix s = Eigen::MatrixXd::Zero(num_constraint_, 1);
  Matrix M = Eigen::MatrixXd::Zero(num_param_, num_param_);
  Matrix iMg = Eigen::MatrixXd::Zero(num_constraint_, 1);
  Matrix iMg_hat = Eigen::MatrixXd::Zero(num_constraint_, 1);
  Matrix u = Eigen::MatrixXd::Zero(num_constraint_, 1);
  Matrix um = Eigen::MatrixXd::Zero(num_constraint_, 1);
  Matrix z = Eigen::MatrixXd::Zero(num_param_, 1);
  Matrix sold = Eigen::MatrixXd::Zero(num_constraint_, 1);
  double coe = static_cast<double>(sqrt(num_constraint_));
  double factor = static_cast<double>(sqrt(num_param_));
  Matrix iMgm = Eigen::MatrixXd::Zero(num_constraint_, 1);
  Matrix snew = Eigen::MatrixXd::Zero(num_constraint_, 1);
  Matrix res = Eigen::MatrixXd::Zero(num_constraint_, 1);

  M = kernel_matrix_ + rho * Matrix_constrain.transpose() * Matrix_constrain;

  while (i < max_iteration_ && flag) {
    z = -M.lu().solve(offset_ + Matrix_constrain.transpose() * (rho * (u - s)));
    iMg = Matrix_constrain * z;
    sold = s;
    iMg_hat = beta * iMg + (1 - beta) * sold;
    s = iMg_hat + u;

    for (unsigned int j = 0; j < s.rows(); j++) {
      s(j, 0) =
          s(j, 0) < Matrix_boundary(j, 0) ? s(j, 0) : Matrix_boundary(j, 0);
    }

    u = u + iMg_hat - s;

    res = iMg - s;
    resz = res.norm();
    iMgm = -iMg;
    snew = -rho * (s - sold);
    resx = snew.norm();
    epspri = factor * termination_tolerance_ +
             termination_tolerance_ * 0.1 *
                 (iMgm.norm() > s.norm() ? iMgm.norm() : s.norm());
    um = rho * u;
    epsdual = coe * termination_tolerance_ +
              0.01 * termination_tolerance_ * um.norm();

    if (resz < epspri && resx < epsdual) {
      flag = 0;
    } else {
      i = i + 1;
    }
  }

  if (i > max_iteration_) {
    AERROR << "ADMM solver failed due to reached max iteration";
    std::stringstream ss;
    ss << "ADMM inputs: " << std::endl;
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

void ADMMQpSolver::set_max_iteration(const int max_iter) {
  max_iteration_ = max_iter;
}
int ADMMQpSolver::max_iteration() const {
  return max_iteration_;
}

// pure virtual
bool ADMMQpSolver::sanity_check() {
  return kernel_matrix_.rows() == kernel_matrix_.cols() &&
         kernel_matrix_.rows() == affine_inequality_matrix_.cols() &&
         kernel_matrix_.rows() == affine_equality_matrix_.cols() &&
         affine_equality_matrix_.rows() == affine_equality_boundary_.rows() &&
         affine_inequality_matrix_.rows() == affine_inequality_boundary_.rows();
}

}  // namespace math
}  // namespace common
}  // namespace roadstar
