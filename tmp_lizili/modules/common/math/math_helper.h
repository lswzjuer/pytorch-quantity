#ifndef MODULES_COMMON_MATH_MATH_HELPER_H_
#define MODULES_COMMON_MATH_MATH_HELPER_H_

#include "Eigen/Core"
#include "Eigen/Dense"

#include "modules/common/log.h"
#include "modules/common/proto/math.pb.h"

namespace roadstar {
namespace common {
namespace math {

void Serialize(const Eigen::Vector2d& vec, Vector2d* const output);

void Serialize(const Eigen::Vector3d& vec, Vector3d* const output);

void Serialize(const Eigen::MatrixXd& vec, MatrixXd* const output);

template <int Rows, int Cols>
void Serialize(const Eigen::Matrix<double, Rows, Cols>& matrix,
               MatrixXd* const output) {
  output->clear_data();
  output->set_rows(matrix.rows());
  output->set_cols(matrix.cols());
  for (int i = 0; i < matrix.cols(); ++i) {
    for (int j = 0; j < matrix.rows(); ++j) {
      output->add_data(matrix(j, i));
    }
  }
}

void Deserialize(const Vector2d& vec, Eigen::Vector2d* const output);

void Deserialize(const Vector3d& vec, Eigen::Vector3d* const output);

void Deserialize(const MatrixXd& matrix, Eigen::MatrixXd* const output);

template <int Rows, int Cols>
void Deserialize(const MatrixXd& matrix,
                 Eigen::Matrix<double, Rows, Cols>* const output) {
  CHECK_EQ(matrix.rows(), Rows);
  CHECK_EQ(matrix.cols(), Cols);
  CHECK_EQ(matrix.data_size(), matrix.rows() * matrix.cols());

  for (int i = 0; i < matrix.cols(); ++i) {
    for (int j = 0; j < matrix.rows(); ++j) {
      (*output)(j, i) = matrix.data(i * matrix.rows() + j);
    }
  }
}

}  // namespace math
}  // namespace common
}  // namespace roadstar

#endif  // MODULES_COMMON_MATH_MATH_HELPER_H_
