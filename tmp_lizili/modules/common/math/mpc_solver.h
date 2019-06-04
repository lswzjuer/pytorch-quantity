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
 * @file mpc_solver.h
 * @brief Convert mpc problem to qp based problem and solve.
 */

#ifndef MODULES_COMMON_MATH_MPC_SOLVER_H_
#define MODULES_COMMON_MATH_MPC_SOLVER_H_

#include <string>
#include <vector>
#include "Eigen/Core"

/**
 * @namespace roadstar::common::math
 * @brief roadstar::common::math
 */

namespace roadstar {
namespace common {
namespace math {
/**
 * @brief Solver for discrete-time model predictive control problem.
 * @param matrix_a The system dynamic matrix
 * @param matrix_b The control matrix
 * @param matrix_c The output matrix
 * @param matrix_q The cost matrix for control state
 * @param matrix_r The cost matrix for control input
 * @param matrix_r_alt The cost matrix for control input
 * @param matrix_u_lower The lower bound control signal constrain matrix
 * @param matrix_u_upper The upper bound control signal constrain matrix
 * @param matrix_y_lower The lower bound output constrain matrix
 * @param matrix_y_upper The upper bound output constrain matrix
 * @param matrix_initial_state The initial state matrix
 * @param reference The control reference vector with respect to time
 * @param eps The control convergence tolerance
 * @param max_iter The maximum iterations for solving ARE
 * @param control The feedback control matrix (pointer)
 */

bool SolveLinearMPC(const Eigen::MatrixXd &matrix_a,
                    const Eigen::MatrixXd &matrix_b,
                    const Eigen::MatrixXd &matrix_c,
                    const Eigen::MatrixXd &matrix_q_augment,
                    const Eigen::MatrixXd &matrix_r_u_augment,
                    const Eigen::MatrixXd &matrix_r_integral_u_augment,
                    const Eigen::MatrixXd &matrix_u_lower,
                    const Eigen::MatrixXd &matrix_u_upper,
                    const Eigen::MatrixXd &matrix_y_lower,
                    const Eigen::MatrixXd &matrix_y_upper,
                    const Eigen::MatrixXd &matrix_initial_state,
                    const std::vector<Eigen::MatrixXd> &reference,
                    const double eps, const int max_iter,
                    const std::string qp_solver_type,
                    std::vector<Eigen::MatrixXd> *control);

bool SolveLinearMPC(const Eigen::MatrixXd &matrix_a,
                    const Eigen::MatrixXd &matrix_b,
                    const Eigen::MatrixXd &matrix_c,
                    const Eigen::MatrixXd &matrix_q_augment,
                    const Eigen::MatrixXd &matrix_r_u_augment,
                    const Eigen::MatrixXd &matrix_r_integral_u_augment,
                    const Eigen::MatrixXd &matrix_u_lower,
                    const Eigen::MatrixXd &matrix_u_upper,
                    const Eigen::MatrixXd &matrix_initial_state,
                    const std::vector<Eigen::MatrixXd> &reference,
                    const double eps, const int max_iter,
                    const std::string qp_solver_type,
                    std::vector<Eigen::MatrixXd> *control);

void AssembleInputConstraint(const unsigned int control_horizon,
                             const unsigned int num_input,
                             const Eigen::MatrixXd &matrix_u,
                             Eigen::MatrixXd *matrix_u_boundary,
                             Eigen::MatrixXd *matrix_u_inequality_constrain);

void AssembleOutputConstraint(
    const unsigned int prediction_horizon, const unsigned int control_horizon,
    const Eigen::MatrixXd &matrix_y, const Eigen::MatrixXd &matrix_b,
    const Eigen::MatrixXd &matrix_c,
    const std::vector<Eigen::MatrixXd> &matrix_a_power,
    const Eigen::MatrixXd &matrix_initial_state,
    Eigen::MatrixXd *matrix_y_boundary,
    Eigen::MatrixXd *matrix_y_inequality_constrain);

}  // namespace math
}  // namespace common
}  // namespace roadstar

#endif  // MODULES_COMMON_MATH_MPC_SOLVER_H_
