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
 * @file
 * @brief Some util functions.
 */

#ifndef MODULES_COMMON_UTIL_H_
#define MODULES_COMMON_UTIL_H_

#include <algorithm>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "google/protobuf/util/message_differencer.h"

#include "modules/common/log.h"
#include "modules/common/math/vec2d.h"
#include "modules/common/proto/geometry.pb.h"
#include "modules/common/proto/pnc_point.pb.h"

/**
 * @namespace roadstar::common::util
 * @brief roadstar::common::util
 */
namespace roadstar {
namespace common {
namespace util {

template <class ForwardIt, class UnaryPredicate>
ForwardIt remove_if(ForwardIt first, ForwardIt last,  // NOLINT
                    UnaryPredicate p) {
  first = std::find_if(first, last, p);
  if (first != last) {
    for (ForwardIt i = first; ++i != last;) {
      if (!p(*i)) *first++ = std::move(*i);
    }
  }
  return first;
}

/**
 * @brief Removes all elements satifying the criteria from the range [first,
 * last) and return the past-the-end iterator for the new end of the range.
 * Unlike remove_if, relative order of the elements that remain is NOT
 * preserved.
 * @param first, last The range of elements to process.
 * @p unary predicate which returns true if the element should remove.
 *
 * */
template <typename ForwardIt, class UnaryPredicate>
ForwardIt unstable_remove_if(ForwardIt first, ForwardIt last,  // NOLINT
                             UnaryPredicate p) {
  for (;;) {
    first = std::find_if(first, last, p);
    while (last != first && p(*(--last))) {
    }
    if (first == last) break;
    *first++ = std::move(*last);
  }
  return last;
}

template <typename T, typename... Args>
[[deprecated]] std::unique_ptr<T> make_unique(Args &&... args) {  // NOLINT
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

template <typename ProtoA, typename ProtoB>
bool IsProtoEqual(const ProtoA &a, const ProtoB &b) {
  return google::protobuf::util::MessageDifferencer::Equals(a, b);
}

struct PairHash {
  template <typename T, typename U>
  size_t operator()(const std::pair<T, U> &pair) const {
    return std::hash<T>()(pair.first) ^ std::hash<U>()(pair.second);
  }
};

/**
 * @brief create a SL point
 * @param s the s value
 * @param l the l value
 * @return a SLPoint instance
 */
SLPoint MakeSLPoint(const double s, const double l);

PointENU MakePointENU(const double x, const double y, const double z);

PointENU MakePointENU(const math::Vec2d &xy);

// roadstar::perception::Point MakePerceptionPoint(const double x, const double
// y,
// const double z);

SpeedPoint MakeSpeedPoint(const double s, const double t, const double v,
                          const double a, const double da);

PathPoint MakePathPoint(const double x, const double y, const double z,
                        const double theta, const double kappa,
                        const double dkappa, const double ddkappa);

template <typename Container>
auto MaxElement(const Container &elements) {
  return *std::max_element(elements.begin(), elements.end());
}

template <typename Container>
auto MinElement(const Container &elements) {
  return *std::min_element(elements.begin(), elements.end());
}

/**
 * @brief calculate the distance beteween PathPoint a and PathPoint b
 * @param a one path point
 * @param b another path point
 * @return sqrt((a.x-b.x)^2 + (a.y-b.y)^2), i.e., the Euclid distance on XY
 * dimension
 */
template <class PointA, class PointB>
inline auto Distance2D(const PointA &a, const PointB &b) {
  return std::hypot(a.x() - b.x(), a.y() - b.y());
}

template <class NumA, class NumB>
inline auto Distance2D(const NumA &x1, const NumA &y1, const NumB &x2,
                       const NumB &y2) {
  return std::hypot(x1 - x2, y1 - y2);
}

template <class Point, class Num>
inline bool PointInRange(
    const Point &a, const Point &b,
    const typename std::enable_if<std::is_arithmetic<Num>::value, Num>::type
        &radius) {
  return (a.x() - b.x()) * (a.x() - b.x()) + (a.y() - b.y()) * (a.y() - b.y()) <
         radius * radius;
}

template <class Num>
inline bool PointInRange(const Num &x1, const Num &x2, const Num &y1,
                         const Num &y2, const Num &radius) {
  return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) < radius * radius;
}

template <class Point>
inline auto MidPoint2D(const Point &a, const Point &b) {
  using DistanceType = decltype(Distance2D(a, a));
  return std::make_tuple((a.x() + b.x()) / DistanceType(2),
                         (a.y() + b.y()) / DistanceType(2));
}

/* @brief Determine whether the point is inside the given polygon
 * @param x x-coordinate of the given point
 * @param y y-coordinate of the given point
 * @param begin Begin iterator of the polygon
 * @param end End iterator of the polygon
 * @return the Result
 */
template <class PointIter>
bool PointInPolygon(const decltype(std::declval<PointIter>()->x()) &x,
                    const decltype(std::declval<PointIter>()->x()) &y,
                    PointIter begin, PointIter end) {
  auto cross_line = [&x, &y](PointIter &point1, PointIter &point2) -> bool {
    auto x1 = point1->x(), x2 = point2->x(), y1 = point1->y(), y2 = point2->y();
    if (y1 == y2) {
      return false;
    } else if (y1 > y2) {
      std::swap(x1, x2), std::swap(y1, y2);
    }
    if (y < y1 || y > y2) return false;
    return (x2 - x1) * (y - y2) - (x - x2) * (y2 - y1) > 0;
  };
  PointIter last = begin, i = begin;
  unsigned count = 0;
  for (++i; i != end; ++i, ++last) {
    if (cross_line(last, i)) ++count;
  }
  if (cross_line(begin, last)) ++count;
  return (count & 1);
}

/* @brief Return the nearest point (in 2 dimension) of the point set to the
 * given point
 * @param point The given point
 * @param begin The begin iterator of the point set
 * @param end The end iterator of the point set
 * @return The iterator of the nearest point
 */
template <class Point, class PointIter>
auto NearestPoint2D(const Point &point, PointIter begin, PointIter end) {
  return std::min_element(
      begin, end,
      [&point](typename std::iterator_traits<PointIter>::value_type const &a,
               typename std::iterator_traits<PointIter>::value_type const &b) {
        return Distance2D(point, a) < Distance2D(point, b);
      });
}

/* @brief Return the fre`chet distance of two given polygon line
 * @param begin The begin iterator of the first polygon line
 * @param end The begin iterator of the first polygon line
 * @param begin The begin iterator of the second polygon line
 * @param end The begin iterator of the second polygon line
 * @return The fre`chet distance
 */
template <class IterA, class IterB>
auto DiscreteFrechetDistance(IterA a_begin, IterA a_end, IterB b_begin,
                             IterB b_end) {
  using DistanceType = decltype(Distance2D(*a_begin, *b_begin));
  std::vector<std::vector<DistanceType>> ca;
  size_t i = 0u, j = 0u;
  if (a_begin == a_end || b_begin == b_end) {
    return std::numeric_limits<DistanceType>::infinity();
  }
  for (auto a = a_begin; a != a_end; ++a, ++i) {
    j = 0u;
    ca.resize(i + 1);
    for (auto b = b_begin; b != b_end; ++b, ++j) {
      ca[i].resize(j + 1);
      if (i == 0u && j == 0u) {
        ca[i][j] = Distance2D(*a, *b);
      } else if (i == 0u) {
        ca[i][j] = std::max(ca[i][j - 1], Distance2D(*a, *b));
      } else if (j == 0u) {
        ca[i][j] = std::max(ca[i - 1][j], Distance2D(*a, *b));
      } else {
        ca[i][j] =
            std::max(std::min({ca[i - 1][j - 1], ca[i][j - 1], ca[i - 1][j]}),
                     Distance2D(*a, *b));
      }
    }
  }
  return ca[i - 1][j - 1];
}

template <class PointA, class PointB>
auto ProjectPointOnSegment(const PointA &p, const PointB &end_point1,
                           const PointB &end_point2,
                           bool allow_outlier = false) {
  const auto &x0 = p.x(), &x1 = end_point1.x(), &x2 = end_point2.x(),
             &y0 = p.y(), &y1 = end_point1.y(), &y2 = end_point2.y();
  using DistanceType = decltype(Distance2D(p, p));

  DistanceType u = (x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1);

  auto u_denom = std::pow(x2 - x1, 2) + std::pow(y2 - y1, 2);

  if (u_denom != 0) {
    u /= u_denom;

    DistanceType x = x1 + (u * (x2 - x1));
    DistanceType y = y1 + (u * (y2 - y1));

    auto min_x = std::min(x1, x2);
    auto max_x = std::max(x1, x2);

    auto min_y = std::min(y1, y2);
    auto max_y = std::max(y1, y2);

    if (allow_outlier ||
        (x >= min_x && x <= max_x && y >= min_y && y <= max_y)) {
      return std::make_tuple(x, y);
    }
  }
  return std::make_tuple(std::numeric_limits<DistanceType>::quiet_NaN(),
                         std::numeric_limits<DistanceType>::quiet_NaN());
}

template <class Point, class PointIter>
auto ProjectPointOnCurve(Point p, PointIter begin, PointIter end,
                         bool allow_outlier = false) {
  using DistanceType = decltype(Distance2D(p, p));
  DistanceType min_dist = std::numeric_limits<DistanceType>::max();
  DistanceType min_x, min_y;
  min_x = min_y = std::numeric_limits<DistanceType>::quiet_NaN();
  for (auto a = begin, b = std::next(a); a != end && b != end; ++a, ++b) {
    auto [x, y] = ProjectPointOnSegment(p, *a, *b, allow_outlier);
    if (std::isnan(x) || std::isnan(y)) continue;
    auto dist = Distance2D(p.x(), p.y(), x, y);
    if (dist < min_dist) {
      min_dist = dist;
      min_x = x, min_y = y;
    }
  }
  return std::make_tuple(min_x, min_y);
}

template <class Point, class PointIter>
auto DistanceToCurve(Point p, PointIter begin, PointIter end,
                     bool allow_outlier = false) {
  using DistanceType = decltype(Distance2D(p, p));
  auto [x, y] = ProjectPointOnCurve(p, begin, end, allow_outlier);
  if (std::isnan(x) || std::isnan(y)) {
    return std::numeric_limits<DistanceType>::quiet_NaN();
  }
  return Distance2D(x, y, p.x(), p.y());
}

bool GetCommandOutput(const std::string &command, std::string *result);

template <ssize_t N, class... Args>
std::ostream &PrintTuple(std::ostream &os, const std::tuple<Args...> &t) {
  if constexpr (N > 1) {
    return PrintTuple<N - 1, Args...>(os, t) << ", " << std::get<N - 1>(t);
  } else if constexpr (N == 1) {
    return os << std::get<N - 1>(t);
  } else {
    return os;
  }
}

/**
 * uniformly slice a segment [start, end] to num + 1 pieces
 * the result sliced will contain the n + 1 points that slices the provided
 * segment. `start` and `end` will be the first and last element in `sliced`.
 */
template <typename T>
void UniformSlice(const T start, const T end, uint32_t num,
                  std::vector<T> *sliced) {
  if (!sliced || num == 0) {
    return;
  }
  const T delta = (end - start) / num;
  sliced->resize(num + 1);
  T s = start;
  for (uint32_t i = 0; i < num; ++i, s += delta) {
    sliced->at(i) = s;
  }
  sliced->at(num) = end;
}

}  // namespace util
}  // namespace common
}  // namespace roadstar

template <typename A, typename B>
std::ostream &operator<<(std::ostream &os, const std::pair<A, B> &p) {
  return os << "first: " << p.first << ", second: " << p.second;
}

template <class... Args>
std::ostream &operator<<(std::ostream &os, const std::tuple<Args...> &t) {
  os << "(";
  return roadstar::common::util::PrintTuple<sizeof...(Args), Args...>(os, t)
         << ")";
}

#endif  // MODULES_COMMON_UTIL_H_
