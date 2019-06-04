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

#include "modules/common/math/mpc_solver.h"

#include "gtest/gtest.h"

namespace roadstar {
namespace common {
namespace math {

class MPCCASE1 : public ::testing::Test {
 protected:
  const unsigned int STATES_ = 4;
  const unsigned int CONTROLS_ = 2;
  const unsigned int HORIZON_ = 10;
  const int MAX_ITER_ = 1000;
  const double EPS_ = 1e-6;

  Eigen::MatrixXd A_, B_, C_, Q_, R_u_, R_integral_u_;
  Eigen::MatrixXd Q_augment_, R_u_augment_, R_integral_u_augment_;
  Eigen::MatrixXd lower_bound_, upper_bound_;
  Eigen::MatrixXd initial_state_, reference_state_;

  std::vector<Eigen::MatrixXd> reference_;
  Eigen::MatrixXd control_matrix_;
  std::vector<Eigen::MatrixXd> control_;

  virtual void SetUp() {
    A_ = Eigen::MatrixXd::Zero(STATES_, STATES_);
    A_ << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1;

    B_ = Eigen::MatrixXd::Zero(STATES_, CONTROLS_);
    B_ << 0, 1, 0, 0, 1, 0, 0, 1;

    C_ = Eigen::MatrixXd::Identity(STATES_, STATES_);

    Q_ = Eigen::MatrixXd::Zero(STATES_, STATES_);
    Q_ << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

    R_u_ = Eigen::MatrixXd::Zero(CONTROLS_, CONTROLS_);
    R_u_ << 1, 0, 0, 1;

    R_integral_u_ = Eigen::MatrixXd::Zero(CONTROLS_, CONTROLS_);
    R_integral_u_ << 0, 0, 0, 0;

    Q_augment_ = Eigen::MatrixXd::Zero(B_.rows() * HORIZON_, B_.rows() * HORIZON_);
    R_u_augment_ = Eigen::MatrixXd::Zero(B_.cols() * HORIZON_, B_.cols() * HORIZON_);
    R_integral_u_augment_ =
        Eigen::MatrixXd::Zero(B_.cols() * HORIZON_, B_.cols() * HORIZON_);

    for (unsigned int i = 0; i < HORIZON_; ++i) {
      Q_augment_.block(i * Q_.rows(), i * Q_.rows(), Q_.rows(), Q_.rows()) = Q_;
      R_u_augment_.block(i * R_u_.rows(), i * R_u_.rows(), R_u_.cols(),
                        R_u_.cols()) = R_u_;
      R_integral_u_augment_.block(i * R_integral_u_.rows(),
                                 i * R_integral_u_.rows(), R_integral_u_.cols(),
                                 R_integral_u_.cols()) = R_integral_u_;
    }

    lower_bound_ = Eigen::MatrixXd::Zero(CONTROLS_, 1);
    lower_bound_ << -10, -10;

    upper_bound_ = Eigen::MatrixXd::Zero(CONTROLS_, 1);
    upper_bound_ << 10, 10;

    initial_state_ = Eigen::MatrixXd::Zero(STATES_, 1);
    initial_state_ << 0, 0, 0, 0;

    reference_state_ = Eigen::MatrixXd::Zero(STATES_, 1);
    reference_state_ << 200, 200, 0, 0;

    reference_.resize(HORIZON_, reference_state_);

    control_matrix_ = Eigen::MatrixXd::Zero(CONTROLS_, 1);
    control_matrix_ << 0, 0;

    control_.resize(HORIZON_, control_matrix_);
    for (size_t i = 0; i < control_.size(); ++i) {
      for (size_t i = 1; i < control_.size(); ++i) {
        control_[i - 1] = control_[i];
      }
      control_[HORIZON_ - 1] = control_matrix_;
    }
  }
};

class MPCCASE2 : public ::testing::Test {
 protected:
  const unsigned int STATES_ = 4;
  const unsigned int CONTROLS_ = 1;
  const unsigned int HORIZON_ = 10;
  const int MAX_ITER_ = 1000;
  const double EPS_ = 1e-6;

  Eigen::MatrixXd A_, B_, C_, Q_, R_u_, R_integral_u_;
  Eigen::MatrixXd Q_augment_, R_u_augment_, R_integral_u_augment_;
  Eigen::MatrixXd lower_bound_, upper_bound_;
  Eigen::MatrixXd initial_state_, reference_state_;

  std::vector<Eigen::MatrixXd> reference_;
  Eigen::MatrixXd control_matrix_;
  std::vector<Eigen::MatrixXd> control_;

  virtual void SetUp() {
    A_ = Eigen::MatrixXd::Zero(STATES_, STATES_);
    A_ << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1;

    B_ = Eigen::MatrixXd::Zero(STATES_, CONTROLS_);
    B_ << 0, 0, 1, 0;

    C_ = Eigen::MatrixXd::Identity(STATES_, STATES_);

    Q_ = Eigen::MatrixXd::Zero(STATES_, STATES_);
    Q_ << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

    R_u_ = Eigen::MatrixXd::Zero(CONTROLS_, CONTROLS_);
    R_u_ << 1;

    R_integral_u_ = Eigen::MatrixXd::Zero(CONTROLS_, CONTROLS_);
    R_integral_u_ << 0;

    Q_augment_ = Eigen::MatrixXd::Zero(B_.rows() * HORIZON_, B_.rows() * HORIZON_);
    R_u_augment_ = Eigen::MatrixXd::Zero(B_.cols() * HORIZON_, B_.cols() * HORIZON_);
    R_integral_u_augment_ =
        Eigen::MatrixXd::Zero(B_.cols() * HORIZON_, B_.cols() * HORIZON_);

    for (unsigned int i = 0; i < HORIZON_; ++i) {
      Q_augment_.block(i * Q_.rows(), i * Q_.rows(), Q_.rows(), Q_.rows()) = Q_;
      R_u_augment_.block(i * R_u_.rows(), i * R_u_.rows(), R_u_.cols(),
                        R_u_.cols()) = R_u_;
      R_integral_u_augment_.block(i * R_integral_u_.rows(),
                                 i * R_integral_u_.rows(), R_integral_u_.cols(),
                                 R_integral_u_.cols()) = R_integral_u_;
    }

    lower_bound_ = Eigen::MatrixXd::Zero(CONTROLS_, 1);
    lower_bound_ << -10;

    upper_bound_ = Eigen::MatrixXd::Zero(CONTROLS_, 1);
    upper_bound_ << 10;

    initial_state_ = Eigen::MatrixXd::Zero(STATES_, 1);
    initial_state_ << 30, 30, 0, 0;

    reference_state_ = Eigen::MatrixXd::Zero(STATES_, 1);
    reference_state_ << 30, 30, 0, 0;

    reference_.resize(HORIZON_, reference_state_);

    control_matrix_ = Eigen::MatrixXd::Zero(CONTROLS_, 1);
    control_matrix_ << 0;

    control_.resize(HORIZON_, control_matrix_);
    for (size_t i = 0; i < control_.size(); ++i) {
      for (size_t i = 1; i < control_.size(); ++i) {
        control_[i - 1] = control_[i];
      }
      control_[HORIZON_ - 1] = control_matrix_;
    }
  }
};

TEST_F(MPCCASE1, ACTIVESET) {
  SolveLinearMPC(A_, B_, C_, Q_augment_, R_u_augment_, R_integral_u_augment_,
                 lower_bound_, upper_bound_, initial_state_, reference_, EPS_,
                 MAX_ITER_, "ACTIVE_SET", &control_);
  EXPECT_FLOAT_EQ(upper_bound_(0), control_[0](0));
}

TEST_F(MPCCASE1, GPAD) {
  SolveLinearMPC(A_, B_, C_, Q_augment_, R_u_augment_, R_integral_u_augment_,
                 lower_bound_, upper_bound_, initial_state_, reference_, EPS_,
                 MAX_ITER_, "GPAD", &control_);
  EXPECT_FLOAT_EQ(upper_bound_(0), control_[0](0));
}

TEST_F(MPCCASE1, ADMM) {
  SolveLinearMPC(A_, B_, C_, Q_augment_, R_u_augment_, R_integral_u_augment_,
                 lower_bound_, upper_bound_, initial_state_, reference_, EPS_,
                 MAX_ITER_, "ADMM", &control_);
  EXPECT_FLOAT_EQ(upper_bound_(0), control_[0](0));
}

TEST_F(MPCCASE2, ACTIVESET) {
  SolveLinearMPC(A_, B_, C_, Q_augment_, R_u_augment_, R_integral_u_augment_,
                 lower_bound_, upper_bound_, initial_state_, reference_, EPS_,
                 MAX_ITER_, "ACTIVE_SET", &control_);
  EXPECT_NEAR(0.0, control_[0](0), 1e-7);
}

TEST_F(MPCCASE2, GPAD) {
  SolveLinearMPC(A_, B_, C_, Q_augment_, R_u_augment_, R_integral_u_augment_,
                 lower_bound_, upper_bound_, initial_state_, reference_, EPS_,
                 MAX_ITER_, "GPAD", &control_);
  EXPECT_NEAR(0.0, control_[0](0), 1e-7);
}

TEST_F(MPCCASE2, ADMM) {
  SolveLinearMPC(A_, B_, C_, Q_augment_, R_u_augment_, R_integral_u_augment_,
                 lower_bound_, upper_bound_, initial_state_, reference_, EPS_,
                 MAX_ITER_, "ADMM", &control_);
  EXPECT_NEAR(0.0, control_[0](0), 1e-7);
}

}  // namespace math
}  // namespace common
}  // namespace roadstar
