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
#ifndef MODULES_COMMON_MATH_GRID_H_
#define MODULES_COMMON_MATH_GRID_H_

#include <algorithm>
#include <array>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "modules/common/log.h"
#include "modules/common/util/string_util.h"

#include "modules/common/math/aabox2d.h"
#include "modules/common/math/box2d.h"
#include "modules/common/math/math_utils.h"

namespace roadstar {
namespace common {
namespace math {
/**
 * @class Grid
 * @brief Implements a class of grid with 2-dimension.
 * @brief and it can save complex structure.
 * @brief row major.
 */
template <typename T>
class Grid2d {
 public:
  /**
   * @brief Default constructor.
   * Creates 0 grid.
   */
  Grid2d() : sizes_({0, 0}), scale_({1, 1}), offset_({0, 0}) {
    values_ptr_.reset(new std::vector<T>(sizes_[0] * sizes_[1]));
    size_ = sizes_[0] * sizes_[1];
  }

  explicit Grid2d(const std::array<int, 2> size)
      : sizes_(size), scale_({1, 1}), offset_({0, 0}) {
    CHECK_GT(static_cast<double>(size[0]), -kMathEpsilon);
    CHECK_GT(static_cast<double>(size[1]), -kMathEpsilon);
    values_ptr_.reset(new std::vector<T>(size[0] * size[1]));
    size_ = sizes_[0] * sizes_[1];
  }

  explicit Grid2d(const std::array<int, 2> size,
                  const std::array<double, 2> scale,
                  const std::array<double, 2> offset)
      : sizes_(size), scale_(scale), offset_(offset) {
    CHECK_GT(static_cast<double>(size[0]), -kMathEpsilon);
    CHECK_GT(static_cast<double>(size[1]), -kMathEpsilon);
    values_ptr_.reset(new std::vector<T>(size[0] * size[1]));
    size_ = sizes_[0] * sizes_[1];
  }

  T& operator()(const int index) {
    return (*values_ptr_)[index];
  }

  const T& operator()(const int index) const {
    return (*values_ptr_)[index];
  }

  T& operator()(const int row, const int col) {
    return (*values_ptr_)[Coord2Index(row, col)];
  }

  const T& operator()(const int row, const int col) const {
    return (*values_ptr_)[Coord2Index(row, col)];
  }

  template <typename CoordT>
  T& GetGrid(const CoordT coord_x, const CoordT coord_y) {
    const auto grid_coord = GetGridCoord(coord_x, coord_y);
    return (*this)(grid_coord[0], grid_coord[1]);
  }

  template <typename CoordT>
  const T& GetGrid(const CoordT coord_x, const CoordT coord_y) const {
    const auto grid_coord = GetGridCoord(coord_x, coord_y);
    return (*this)(grid_coord[0], grid_coord[1]);
  }

  int row() const {
    return sizes_[0];
  }

  int col() const {
    return sizes_[1];
  }

  int size() const {
    return size_;
  }

  std::array<double, 2> scale() const {
    return scale_;
  }

  std::array<double, 2> offset() const {
    return offset_;
  }

  /**
   * @brief judge if point in the grid.
   * @param : 2 X-Y coords
   * @return if point in the grid
   */

  bool IsPointInGrid(const double coord_x, const double coord_y) const {
    std::array<int, 2> index = GetGridCoord(coord_x, coord_y);
    return index[0] >= 0 && index[0] < sizes_[0] && index[1] >= 0 &&
           index[1] < sizes_[1];
  }

  /**
   * @brief return array of coordinate of grid cover aabox.
   * @param box: AABox
   * @return coordinate array
   */

  std::vector<std::array<int, 2>> GetOverlapArea(const AABox2d& box) const {
    double min_x, max_x, min_y, max_y;
    min_x = box.min_x();
    max_x = box.max_x();
    min_y = box.min_y();
    max_y = box.max_y();
    std::vector<std::array<int, 2>> coords;
    std::array<int, 2> start = GetGridCoord(min_x, min_y);
    std::array<int, 2> end = GetGridCoord(max_x, max_y);
    for (int i = 0; i < 2; i++) {
      end[i] = std::min(end[i], sizes_[i] - 1);
      start[i] = std::max(start[i], 0);
      if (start[i] > end[i]) {
        return coords;
      }
    }
    coords.reserve((end[0] - start[0] + 1) * (end[1] - start[1] + 1));
    for (int i = 0; i < end[0] - start[0] + 1; i++) {
      for (int j = 0; j < end[1] - start[1] + 1; j++) {
        std::array<int, 2> coord = {start[0] + i, start[1] + j};
        coords.emplace_back(coord);
      }
    }
    return coords;
  }
  std::vector<std::array<int, 2>> GetOverlapArea(const Box2d& box) const {
    std::vector<std::array<int, 2>> coords;
    AABox2d aabox = box.GetAABox();
    // corners={C0,C1,C2,C3,C0}
    std::vector<Vec2d> corners = box.GetAllCorners();
    corners.push_back(corners[0]);
    double start_y = std::max(aabox.min_y() + scale_[1] / 2, offset_[1]);
    double end_y = std::min(aabox.max_y(), offset_[1] + scale_[1] * sizes_[1]);
    // find edge which x0 is between Ci,Ci+1
    for (double x0 = start_y; x0 < end_y; x0 += scale_[1]) {
      std::vector<double> y;
      for (int i = 0; i < corners.size() - 1; i++) {
        if (x0 <= std::max(corners[i].y(), corners[i + 1].y()) &&
            x0 >= std::min(corners[i].y(), corners[i + 1].y())) {
          if (corners[i + 1].y() != corners[i].y()) {
            y.push_back(
                (corners[i].x() + (x0 - corners[i].y()) *
                                      (corners[i + 1].x() - corners[i].x()) /
                                      (corners[i + 1].y() - corners[i].y())));
          } else {
            y.push_back(corners[i].x());
          }
        }
      }
      // delete the same element
      y.erase(std::unique(y.begin(), y.end()), y.end());
      if (y.size() >= 2) {
        if (y[0] > y[1]) std::swap(y[0], y[1]);
        int row = static_cast<int>((x0 - offset_[0]) / scale_[0]);
        int col0 =
            std::max(static_cast<int>((y[0] - offset_[1]) / scale_[1]), 0);
        int col1 = std::min(static_cast<int>((y[1] - offset_[1]) / scale_[1]),
                            sizes_[1]);
        for (int col = col0; col <= col1; col++) {
          coords.push_back({row, col});
        }
      }
    }
    return coords;
  }

  Grid2d<T> DeepCopy() const {
    Grid2d<T> grid({sizes_[0], sizes_[1]}, {scale_[0], scale_[1]},
                   {offset_[0], offset_[1]});
    for (int i = 0; i < sizes_[0]; i++) {
      for (int j = 0; j < sizes_[1]; j++) {
        (*grid.values_ptr_)[i * sizes_[1] + j] =
            (*values_ptr_)[i * sizes_[1] + j];
      }
    }
    return grid;
  }
  /**
   * @brief given coordinate, map it into coordinate in grid.
   * @param : X-Y coordinate
   * @return coordinate in grid
   */
  std::array<int, 2> GetGridCoord(const double coord_x,
                                  const double coord_y) const {
    return {static_cast<int>((coord_x - offset_[0]) / scale_[0]),
            static_cast<int>((coord_y - offset_[1]) / scale_[1])};
  }

  int GetGridIndex(const double coord_x, const double coord_y) const {
    const auto grid_coord = GetGridCoord(coord_x, coord_y);
    return Coord2Index(grid_coord[0], grid_coord[1]);
  }

  int Coord2Index(const int row, const int col) const {
    return row * sizes_[1] + col;
  }

 private:
  std::shared_ptr<std::vector<T>> values_ptr_;
  std::array<int, 2> sizes_;
  int size_;
  std::array<double, 2> scale_;
  std::array<double, 2> offset_;
};

}  // namespace math
}  // namespace common
}  // namespace roadstar

#endif /*MODULES_COMMON_MATH_GRID_H_*/
