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

#ifndef MODULES_COMMON_MATH_POLYNOMIAL_CURVE_H_
#define MODULES_COMMON_MATH_POLYNOMIAL_CURVE_H_

#include <array>
#include <functional>
#include <iterator>
#include <vector>

#include "Eigen/QR"

namespace roadstar {
namespace common {
namespace math {

template <size_t order = 1, class DataType = double>
class PolynomialCurve {
 public:
  template <class IterX, class IterY>
  void PolyFit(IterX x_begin, IterX x_end, IterY y_begin, IterY y_end) {
    const std::vector<DataType> yv(y_begin, y_end);
    auto y =
        Eigen::Matrix<DataType, Eigen::Dynamic, 1>::Map(&yv.front(), yv.size());
    Eigen::Matrix<DataType, Eigen::Dynamic, order + 1> A(yv.size(), order + 1);
    size_t x_count = 0u;
    for (auto i = x_begin; i != x_end; ++i, ++x_count) {
      for (size_t j = 0u; j < order + 1; j++) {
        A(x_count, j) = j == 0 ? 1 : *i * A(x_count, j - 1);
      }
    }
    assert(x_count == yv.size());
    assert(x_count >= order + 1);

    Eigen::Matrix<DataType, Eigen::Dynamic, 1> result =
        A.householderQr().solve(y);  // Cannot use `auto` here
    for (size_t i = 0; i < order + 1; ++i) {
      coeff_[i] = result[i];
    }
  }

  template <class PointIter>
  using GetFunctionType = std::function<DataType(
      typename std::iterator_traits<PointIter>::value_type const& p)>;

  template <class PointIter>
  void PolyFit(PointIter begin, PointIter end,
               GetFunctionType<PointIter> get_x = [](auto p) { return p.x(); },
               GetFunctionType<PointIter> get_y =
                   [](auto p) { return p.y(); }) {
    std::vector<DataType> xv, yv;
    for (auto p = begin; p != end; ++p) {
      xv.push_back(get_x(*p));
      yv.push_back(get_y(*p));
    }
    PolyFit(xv.begin(), xv.end(), yv.begin(), yv.end());
  }

  DataType PolyEval(DataType x) {
    DataType result = coeff_[0];
    for (size_t i = 1; i < order + 1; ++i, x *= x) {
      result += coeff_[i] * x;
    }
    return result;
  }

  auto& GetCoeff() const {
    return coeff_;
  }

 private:
  std::array<DataType, order + 1> coeff_;
};
}  // namespace math
}  // namespace common
}  // namespace roadstar

#endif  // MODULES_COMMON_MATH_POLYNOMIAL_CURVE_H_
