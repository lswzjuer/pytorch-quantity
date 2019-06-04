#include "modules/common/math/math_helper.h"

#include "gtest/gtest.h"

namespace roadstar {
namespace common {
namespace math {

TEST(MathHelperTest, SerializeVector2d) {
  Eigen::Vector2d vec;
  vec << 1, 2;

  Vector2d output;
  Serialize(vec, &output);
  EXPECT_EQ(output.x(), 1);
  EXPECT_EQ(output.y(), 2);
}

TEST(MathHelperTest, SerializeVector3d) {
  Eigen::Vector3d vec;
  vec << 1, 2, 3;

  Vector3d output;
  Serialize(vec, &output);
  EXPECT_EQ(output.x(), 1);
  EXPECT_EQ(output.y(), 2);
  EXPECT_EQ(output.z(), 3);
}

TEST(MathHelperTest, SerializeMatrixXd) {
  Eigen::MatrixXd matrix;
  matrix.resize(2, 2);
  matrix << 1, 2, 3, 4;

  MatrixXd output;
  Serialize(matrix, &output);
  ASSERT_EQ(output.rows(), 2);
  ASSERT_EQ(output.cols(), 2);

  EXPECT_EQ(output.data(0), 1);
  EXPECT_EQ(output.data(1), 3);
  EXPECT_EQ(output.data(2), 2);
  EXPECT_EQ(output.data(3), 4);
}

TEST(MathHelperTest, SerializeMatrix) {
  Eigen::Matrix2d matrix;
  matrix << 1, 2, 3, 4;

  MatrixXd output;
  Serialize<2, 2>(matrix, &output);
  ASSERT_EQ(output.rows(), 2);
  ASSERT_EQ(output.cols(), 2);

  EXPECT_EQ(output.data(0), 1);
  EXPECT_EQ(output.data(1), 3);
  EXPECT_EQ(output.data(2), 2);
  EXPECT_EQ(output.data(3), 4);
}

TEST(MathHelperTest, DeserializeVector2d) {
  Vector2d vec;
  vec.set_x(1);
  vec.set_y(2);

  Eigen::Vector2d output;
  Deserialize(vec, &output);

  EXPECT_EQ(output(0), 1);
  EXPECT_EQ(output(1), 2);
}

TEST(MathHelperTest, DeserializeVector3d) {
  Vector3d vec;
  vec.set_x(1);
  vec.set_y(2);
  vec.set_z(3);

  Eigen::Vector3d output;
  Deserialize(vec, &output);

  EXPECT_EQ(output(0), 1);
  EXPECT_EQ(output(1), 2);
  EXPECT_EQ(output(2), 3);
}

TEST(MathHelperTest, DeserializeMatrixXd) {
  MatrixXd matrix;
  matrix.add_data(1);
  matrix.add_data(2);
  matrix.add_data(3);
  matrix.add_data(4);
  matrix.set_rows(2);
  matrix.set_cols(2);

  Eigen::MatrixXd output;
  Deserialize(matrix, &output);

  ASSERT_EQ(output.rows(), 2);
  ASSERT_EQ(output.cols(), 2);
  EXPECT_EQ(output(0, 0), 1);
  EXPECT_EQ(output(1, 0), 2);
  EXPECT_EQ(output(0, 1), 3);
  EXPECT_EQ(output(1, 1), 4);
}

TEST(MathHelperTest, DeserializeMatrix) {
  MatrixXd matrix;
  matrix.add_data(1);
  matrix.add_data(2);
  matrix.add_data(3);
  matrix.add_data(4);
  matrix.set_rows(2);
  matrix.set_cols(2);

  Eigen::Matrix2d output;
  Deserialize<2, 2>(matrix, &output);

  EXPECT_EQ(output(0, 0), 1);
  EXPECT_EQ(output(1, 0), 2);
  EXPECT_EQ(output(0, 1), 3);
  EXPECT_EQ(output(1, 1), 4);
}

}  // namespace math
}  // namespace common
}  // namespace roadstar
