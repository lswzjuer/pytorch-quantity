/******************************************************************************
 * Copyright 2019 The Roadstar Authors. All Rights Reserved.
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

#ifndef MODULES_COMMON_MATH_EIGEN_UTIL_H
#define MODULES_COMMON_MATH_EIGEN_UTIL_H

#include <algorithm>
#include <vector>
#include "Eigen/Core"

namespace roadstar {
namespace common {
namespace math {

template <typename T>
using EArrXt = Eigen::Array<T, Eigen::Dynamic, 1>;

template <class Derived, class Derived1, class Derived2>
void GetSubArray(const Eigen::ArrayBase<Derived> &array,
                 const Eigen::ArrayBase<Derived1> &indices,
                 Eigen::ArrayBase<Derived2> *out_array) {
  CHECK_EQ(array.cols(), 1);

  out_array->derived().resize(indices.size());
  for (int i = 0; i < indices.size(); i++) {
    CHECK_LT(indices[i], array.size());
    (*out_array)[i] = array[indices[i]];
  }
}

template <class Derived, class Derived1>
EArrXt<typename Derived::Scalar> GetSubArray(
    const Eigen::ArrayBase<Derived> &array,
    const Eigen::ArrayBase<Derived1> &indices) {
  using T = typename Derived::Scalar;
  EArrXt<T> ret(indices.size());
  GetSubArray(array, indices, &ret);
  return ret;
}

// return 2d sub array of 'array' based on row indices 'row_indices'
template <class Derived, class Derived1, class Derived2>
void GetSubArrayRows(const Eigen::ArrayBase<Derived> &array2d,
                     const Eigen::ArrayBase<Derived1> &row_indices,
                     Eigen::ArrayBase<Derived2> *out_array) {
  out_array->derived().resize(row_indices.size(), array2d.cols());

  for (int i = 0; i < row_indices.size(); i++) {
    CHECK_LT(row_indices[i], array2d.size());
    out_array->row(i) =
        array2d.row(row_indices[i]).template cast<typename Derived2::Scalar>();
  }
}

template <typename T>
Eigen::Map<const EArrXt<T>> AsEArrXt(const std::vector<T> &arr) {
  return {arr.data(), static_cast<int>(arr.size())};
}

template <class Derived>
std::vector<int> GetArrayIndices(const Eigen::ArrayBase<Derived> &array) {
  std::vector<int> ret;
  for (int i = 0; i < array.size(); i++) {
    if (array[i]) {
      ret.push_back(i);
    }
  }
  return ret;
}

}  // namespace math
}  // namespace common
}  // namespace roadstar
#endif  // MODULES_COMMON_MATH_EIGEN_UTIL_H
