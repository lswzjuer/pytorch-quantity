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
#include "modules/common/math/mpc_solver.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include "modules/common/log.h"
#include "modules/common/math/qp_solver/active_set_qp_solver.h"
#include "modules/common/math/qp_solver/admm_qp_solver.h"
#include "modules/common/math/qp_solver/gpad_qp_solver.h"
#include "modules/common/math/qp_solver/qp_solver.h"
#include "modules/common/time/time.h"

namespace roadstar {
namespace common {
namespace math {

using Matrix = Eigen::MatrixXd;

// discrete linear predictive control solver, with control format
// x(i + 1) = A * x(i) + B * u (i)
// y(i + 1) = C * x(i + 1)
bool SolveLinearMPC(const Matrix &matrix_a, const Matrix &matrix_b,
                    const Matrix &matrix_c, const Matrix &matrix_q_augment,
                    const Matrix &matrix_r_u_augment,
                    const Matrix &matrix_r_integral_u_augment,
                    const Matrix &matrix_u_lower, const Matrix &matrix_u_upper,
                    const Matrix &matrix_y_lower, const Matrix &matrix_y_upper,
                    const Matrix &matrix_initial_state,
                    const std::vector<Matrix> &reference, const double eps,
                    const int max_iter, const std::string qp_solver_type,
                    std::vector<Matrix> *control) {
  if (matrix_a.rows() != matrix_a.cols() ||
      matrix_b.rows() != matrix_a.rows() ||
      matrix_c.cols() != matrix_a.rows() ||
      matrix_u_lower.rows() != matrix_u_upper.rows() ||
      matrix_y_lower.rows() != matrix_y_upper.rows() ||
      reference.size() <
          (matrix_r_u_augment.rows() / (*control).at(0).rows())) {
    AERROR << "One or more matrices have incompatible dimensions. Aborting.";
    return false;
  }

  unsigned int prediction_horizon = reference.size();
  unsigned int num_input = (*control).at(0).rows();
  unsigned int control_horizon = matrix_r_u_augment.rows() / num_input;

  // Update augment reference matrix_t
  Matrix matrix_t = Matrix::Zero(matrix_c.rows() * prediction_horizon, 1);
  for (unsigned int j = 0; j < prediction_horizon; ++j) {
    matrix_t.block(j * reference.at(0).size(), 0, reference.at(0).size(), 1) =
        reference.at(j);
  }

  // Update augment control matrix_v
  Matrix matrix_v = Matrix::Zero(num_input * control_horizon, 1);
  for (unsigned int j = 0; j < control_horizon; ++j) {
    matrix_v.block(j * num_input, 0, num_input, 1) = (*control).at(j);
  }

  std::vector<Matrix> matrix_a_power(prediction_horizon + 1);
  matrix_a_power.at(0) = Matrix::Identity(matrix_a.rows(), matrix_a.cols());
  for (size_t i = 1; i < matrix_a_power.size(); ++i) {
    matrix_a_power.at(i) = matrix_a * matrix_a_power.at(i - 1);
  }

  Matrix matrix_k = Matrix::Zero(matrix_c.rows() * prediction_horizon,
                                 matrix_b.cols() * control_horizon);
  for (unsigned int i = 0; i < control_horizon; ++i) {
    matrix_k.block(i * matrix_c.rows(), i * matrix_b.cols(), matrix_c.rows(),
                   matrix_b.cols()) = matrix_c * matrix_b;
  }
  for (unsigned int r = 1; r < control_horizon; ++r) {
    for (unsigned int c = 0; c < r; ++c) {
      matrix_k.block(r * matrix_c.rows(), c * matrix_b.cols(), matrix_c.rows(),
                     matrix_b.cols()) =
          matrix_c * matrix_a_power.at(r - c) * matrix_b;
    }
  }

  for (unsigned int m = control_horizon; m < prediction_horizon; ++m) {
    for (unsigned int n = 0; n < control_horizon; ++n) {
      for (unsigned int p = 0; p < m - n - 1; ++p) {
        matrix_k.block(m * matrix_c.rows(), n * matrix_b.cols(),
                       matrix_c.rows(), matrix_b.cols()) =
            matrix_k.block(m * matrix_c.rows(), n * matrix_b.cols(),
                           matrix_c.rows(), matrix_b.cols()) +
            matrix_c * matrix_a_power.at(m - n - p) * matrix_b;
      }
    }
  }

  // Initialize matrix_k, matrix_m, matrix_t and matrix_v, matrix_qq, matrix_rr,
  // vector of matrix A power
  Matrix matrix_m = Matrix::Zero(matrix_c.rows() * prediction_horizon, 1);
  Matrix matrix_ul = Matrix::Zero(control_horizon * matrix_u_lower.rows(), 1);
  Matrix matrix_uu = Matrix::Zero(control_horizon * matrix_u_upper.rows(), 1);
  Matrix matrix_yl =
      Matrix::Zero(prediction_horizon * matrix_y_lower.rows(), 1);
  Matrix matrix_yu =
      Matrix::Zero(prediction_horizon * matrix_y_upper.rows(), 1);

  Matrix matrix_inequality_constrain_ul =
      Matrix::Zero(control_horizon * matrix_u_lower.rows(),
                   control_horizon * matrix_u_lower.rows());

  Matrix matrix_inequality_constrain_uu =
      Matrix::Zero(control_horizon * matrix_u_upper.rows(),
                   control_horizon * matrix_u_upper.rows());

  Matrix matrix_inequality_constrain_yl = Matrix::Zero(
      prediction_horizon * matrix_y_lower.rows(), control_horizon * num_input);
  Matrix matrix_inequality_constrain_yu = Matrix::Zero(
      prediction_horizon * matrix_y_upper.rows(), control_horizon * num_input);

  // Compute matrix_m
  for (unsigned int i = 0; i < prediction_horizon; ++i) {
    matrix_m.block(i * matrix_c.rows(), 0, matrix_c.rows(), 1) =
        matrix_c * matrix_a_power.at(i + 1) * matrix_initial_state;
  }

  // Compute matrix_ul, matrix_uu, matrix_yl, matrix_yu, matrix_qq, matrix_rr
  AssembleInputConstraint(control_horizon, num_input, matrix_u_lower,
                          &matrix_ul, &matrix_inequality_constrain_ul);
  AssembleInputConstraint(control_horizon, num_input, matrix_u_upper,
                          &matrix_uu, &matrix_inequality_constrain_uu);
  AssembleOutputConstraint(prediction_horizon, control_horizon, matrix_y_lower,
                           matrix_b, matrix_c, matrix_a_power,
                           matrix_initial_state, &matrix_yl,
                           &matrix_inequality_constrain_yl);
  AssembleOutputConstraint(prediction_horizon, control_horizon, matrix_y_upper,
                           matrix_b, matrix_c, matrix_a_power,
                           matrix_initial_state, &matrix_yu,
                           &matrix_inequality_constrain_yu);

  // Compute the coefficient matrix for u
  Matrix matrix_one_u = Matrix::Zero(matrix_b.cols() * control_horizon,
                                     matrix_b.cols() * control_horizon);
  for (unsigned int ur = 0; ur < control_horizon; ++ur) {
    for (unsigned int uc = 0; uc <= ur; ++uc) {
      matrix_one_u.block(ur * matrix_b.cols(), uc * matrix_b.cols(),
                         matrix_b.cols(), matrix_b.cols()) =
          Matrix::Identity(num_input, num_input);
    }
  }

  // Compute the first order coefficient matrix for u
  Matrix matrix_init_u = Matrix::Ones(matrix_b.cols() * control_horizon, 1) *
                         matrix_initial_state(matrix_a.rows() - 1, 0);

  // Update matrix_m1, matrix_m2, convert MPC problem to QP problem done
  Matrix matrix_m1 =
      matrix_k.transpose() * matrix_q_augment * matrix_k + matrix_r_u_augment +
      matrix_one_u.transpose() * matrix_r_integral_u_augment * matrix_one_u;

  Matrix matrix_m2 =
      matrix_k.transpose() * matrix_q_augment * (matrix_m - matrix_t) +
      matrix_one_u.transpose() * matrix_r_integral_u_augment * matrix_init_u;

  // Format in qp_solver
  /**
   * *           min_x  : q(x) = 0.5 * x^T * Q * x  + x^T c
   * *           with respect to:  A * x = b (equality constraint)
   * *                             C * x >= d (inequality constraint)
   * **/

  // TODO(QiL) : change qp solver to box constraint or substitute QPOASES
  // Method 1: QPOASES

  Matrix matrix_inequality_constrain = Matrix::Zero(
      matrix_ul.rows() + matrix_uu.rows() + matrix_yl.rows() + matrix_yu.rows(),
      matrix_ul.rows());
  matrix_inequality_constrain << matrix_inequality_constrain_ul,
      -matrix_inequality_constrain_uu, matrix_inequality_constrain_yl,
      -matrix_inequality_constrain_yu;

  Matrix matrix_inequality_boundary = Matrix::Zero(
      matrix_ul.rows() + matrix_uu.rows() + matrix_yl.rows() + matrix_yu.rows(),
      matrix_ul.cols());
  matrix_inequality_boundary << matrix_ul, -matrix_uu, matrix_yl + matrix_t,
      -matrix_yu - matrix_t;

  Matrix matrix_equality_constrain =
      Matrix::Zero(matrix_ul.rows() + matrix_uu.rows(), matrix_ul.rows());
  Matrix matrix_equality_boundary =
      Matrix::Zero(matrix_ul.rows() + matrix_uu.rows(), matrix_ul.cols());

  // ActiveSetQpSolver or GPADQpSolver or ADMMQpSolver
  std::unique_ptr<QpSolver> qp_solver_ptr;
  bool has_result = false;
  if (qp_solver_type == "ACTIVE_SET") {
    qp_solver_ptr.reset(new ActiveSetQpSolver(
        matrix_m1, matrix_m2, matrix_inequality_constrain,
        matrix_inequality_boundary, matrix_equality_constrain,
        matrix_equality_boundary, max_iter, eps));
    has_result = qp_solver_ptr->Solve();
  }

  if (qp_solver_type == "GPAD") {
    qp_solver_ptr.reset(
        new GPADQpSolver(matrix_m1, matrix_m2, matrix_inequality_constrain,
                         matrix_inequality_boundary, matrix_equality_constrain,
                         matrix_equality_boundary, max_iter, eps));
    has_result = qp_solver_ptr->Solve();
  }

  if (qp_solver_type == "ADMM") {
    qp_solver_ptr.reset(
        new ADMMQpSolver(matrix_m1, matrix_m2, matrix_inequality_constrain,
                         matrix_inequality_boundary, matrix_equality_constrain,
                         matrix_equality_boundary, max_iter, eps));
    has_result = qp_solver_ptr->Solve();
  }

  if (!has_result) {
    AERROR << "Linear MPC solver failed";
    return false;
  }
  matrix_v = qp_solver_ptr->params();
  ADEBUG << "The optimal solution is "
         << 0.5 * matrix_v.transpose() * matrix_m1 * matrix_v +
                matrix_v.transpose() * matrix_m2;

  for (unsigned int i = 0; i < control_horizon; ++i) {
    (*control).at(i) = matrix_v.block(i * num_input, 0, num_input, 1);
  }
  return true;
}

// discrete linear predictive control solver, with control format
// x(i + 1) = A * x(i) + B * u (i)
// y(i + 1) = C * x(i + 1)
bool SolveLinearMPC(const Matrix &matrix_a, const Matrix &matrix_b,
                    const Matrix &matrix_c, const Matrix &matrix_q_augment,
                    const Matrix &matrix_r_u_augment,
                    const Matrix &matrix_r_integral_u_augment,
                    const Matrix &matrix_u_lower, const Matrix &matrix_u_upper,
                    const Matrix &matrix_initial_state,
                    const std::vector<Matrix> &reference, const double eps,
                    const int max_iter, const std::string qp_solver_type,
                    std::vector<Matrix> *control) {
  Matrix matrix_y_lower = -DBL_MAX * Matrix::Ones(matrix_c.rows(), 1);
  Matrix matrix_y_upper = DBL_MAX * Matrix::Ones(matrix_c.rows(), 1);

  return SolveLinearMPC(matrix_a, matrix_b, matrix_c, matrix_q_augment,
                        matrix_r_u_augment, matrix_r_integral_u_augment,
                        matrix_u_lower, matrix_u_upper, matrix_y_lower,
                        matrix_y_upper, matrix_initial_state, reference, eps,
                        max_iter, qp_solver_type, control);
}

void AssembleInputConstraint(const unsigned int control_horizon,
                             const unsigned int num_input,
                             const Eigen::MatrixXd &matrix_u,
                             Eigen::MatrixXd *matrix_u_boundary,
                             Eigen::MatrixXd *matrix_u_inequality_constrain) {
  for (unsigned int i = 0; i < control_horizon; ++i) {
    (*matrix_u_boundary).block(i * num_input, 0, num_input, 1) = matrix_u;
  }

  *matrix_u_inequality_constrain = Matrix::Identity(
      (*matrix_u_boundary).rows(), (*matrix_u_boundary).rows());
}

void AssembleOutputConstraint(
    const unsigned int prediction_horizon, const unsigned int control_horizon,
    const Eigen::MatrixXd &matrix_y, const Eigen::MatrixXd &matrix_b,
    const Eigen::MatrixXd &matrix_c,
    const std::vector<Eigen::MatrixXd> &matrix_a_power,
    const Eigen::MatrixXd &matrix_initial_state,
    Eigen::MatrixXd *matrix_y_boundary,
    Eigen::MatrixXd *matrix_y_inequality_constrain) {
  unsigned int num_output = matrix_c.rows();
  unsigned int num_control = matrix_b.cols();

  *matrix_y_boundary = Matrix::Zero(num_output * prediction_horizon, 1);
  *matrix_y_inequality_constrain = Matrix::Zero(num_output * prediction_horizon,
                                                num_control * control_horizon);

  for (unsigned int i = 0; i < prediction_horizon; ++i) {
    (*matrix_y_boundary).block(i * num_output, 0, num_output, 1) =
        matrix_y - matrix_c * matrix_a_power.at(i + 1) * matrix_initial_state;
  }

  for (unsigned int j = 0; j < control_horizon; ++j) {
    (*matrix_y_inequality_constrain)
        .block(j * num_output, j * num_control, num_output, num_control) =
        matrix_c * matrix_b;
  }

  for (unsigned int i = 1; i < control_horizon; ++i) {
    for (unsigned int j = 0; j < i; ++j) {
      (*matrix_y_inequality_constrain)
          .block(i * num_output, j * num_control, num_output, num_control) =
          matrix_c * matrix_a_power.at(i - j) * matrix_b;
    }
  }
  for (unsigned int k = control_horizon; k < prediction_horizon; ++k) {
    for (unsigned int h = 0; h < control_horizon; ++h) {
      for (unsigned int p = 0; p < k - h - 1; ++p) {
        (*matrix_y_inequality_constrain)
            .block(k * num_output, h * num_control, num_output, num_control) =
            (*matrix_y_inequality_constrain)
                .block(k * num_output, h * num_control, num_output,
                       num_control) +
            matrix_c * matrix_a_power.at(k - h - p) * matrix_b;
      }
    }
  }
}

}  // namespace math
}  // namespace common
}  // namespace roadstar
