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

#include "modules/common/math/polynomial_curve.h"

#include "gtest/gtest.h"

namespace roadstar {
namespace common {
namespace math {

TEST(PolynomialCurveTest, TestPolyFitO1) {
  PolynomialCurve<1, double> curve;
  std::vector<double> xv{1, 2};
  std::vector<double> yv{1, 2};
  curve.PolyFit(xv.begin(), xv.end(), yv.begin(), yv.end());
  auto &coeff = curve.GetCoeff();
  EXPECT_DOUBLE_EQ(coeff[0], 0);
  EXPECT_DOUBLE_EQ(coeff[1], 1);
}

TEST(PolynomialCurveTest, TestPolyFitO2) {
  PolynomialCurve<2, double> curve;
  std::vector<double> xv{0, 1, -1};
  std::vector<double> yv{1, 3, 3};
  curve.PolyFit(xv.begin(), xv.end(), yv.begin(), yv.end());
  auto &coeff = curve.GetCoeff();
  EXPECT_DOUBLE_EQ(coeff[0], 1);
  EXPECT_DOUBLE_EQ(coeff[1], 0);
  EXPECT_DOUBLE_EQ(coeff[2], 2);
  EXPECT_DOUBLE_EQ(curve.PolyEval(3), 19);
}

TEST(PolynomialCurveTest, TestPolyFitP) {
  struct TestPoint {
    double x_;
    double y_;
    inline auto x() {
      return x_;
    }
    inline auto y() {
      return y_;
    }
  };
  PolynomialCurve<1, double> curve;
  std::vector<TestPoint> points{{1, 1}, {2, 2}};
  curve.PolyFit(points.begin(), points.end());
  auto &coeff = curve.GetCoeff();
  EXPECT_DOUBLE_EQ(coeff[0], 0);
  EXPECT_DOUBLE_EQ(coeff[1], 1);
  EXPECT_DOUBLE_EQ(curve.PolyEval(3), 3);
}

TEST(PolynomialCurveTest, TestPolyFitP2) {
  struct TestPoint {
    double x;
    double y;
  };
  PolynomialCurve<1, double> curve;
  std::vector<TestPoint> points{{1, 1}, {2, 2}};
  curve.PolyFit(points.begin(), points.end(), [](auto p) { return p.x; },
                [](auto p) { return p.y; });
  auto &coeff = curve.GetCoeff();
  EXPECT_DOUBLE_EQ(coeff[0], 0);
  EXPECT_DOUBLE_EQ(coeff[1], 1);
  EXPECT_DOUBLE_EQ(curve.PolyEval(3), 3);
}

}  // namespace math
}  // namespace common
}  // namespace roadstar
