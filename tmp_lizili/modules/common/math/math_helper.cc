#include "modules/common/math/math_helper.h"

namespace roadstar {
namespace common {
namespace math {

void Serialize(const Eigen::Vector2d& vec, Vector2d* const output) {
  output->set_x(vec(0));
  output->set_y(vec(1));
}

void Serialize(const Eigen::Vector3d& vec, Vector3d* const output) {
  output->set_x(vec(0));
  output->set_y(vec(1));
  output->set_z(vec(2));
}

void Serialize(const Eigen::MatrixXd& matrix, MatrixXd* const output) {
  output->clear_data();
  output->set_rows(matrix.rows());
  output->set_cols(matrix.cols());
  for (int i = 0; i < matrix.cols(); ++i) {
    for (int j = 0; j < matrix.rows(); ++j) {
      output->add_data(matrix(j, i));
    }
  }
}

void Deserialize(const Vector2d& vec, Eigen::Vector2d* const output) {
  (*output)(0) = vec.x();
  (*output)(1) = vec.y();
}

void Deserialize(const Vector3d& vec, Eigen::Vector3d* const output) {
  (*output)(0) = vec.x();
  (*output)(1) = vec.y();
  (*output)(2) = vec.z();
}

void Deserialize(const MatrixXd& matrix, Eigen::MatrixXd* const output) {
  CHECK_EQ(matrix.data_size(), matrix.rows() * matrix.cols());

  output->resize(matrix.rows(), matrix.cols());
  for (int i = 0; i < matrix.cols(); ++i) {
    for (int j = 0; j < matrix.rows(); ++j) {
      (*output)(j, i) = matrix.data(i * matrix.rows() + j);
    }
  }
}

}  // namespace math
}  // namespace common
}  // namespace roadstar
