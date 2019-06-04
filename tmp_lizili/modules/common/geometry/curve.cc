#include "modules/common/geometry/curve.h"

#include <assert.h>
#include <algorithm>
#include <vector>

namespace roadstar {
namespace common {
namespace geometry {
namespace {

double ComputeDist(double x0, double y0, const Spline& curve_x,
                   const Spline& curve_y, double s) {
  const double x = curve_x(s);
  const double y = curve_y(s);

  return std::hypot(x - x0, y - y0);
}

}  // namespace

void Curve::Clear() {
  curve_x_ = Spline();
  curve_y_ = Spline();
  curve_theta_ = Spline();
  length_ = 0;
}

void Curve::FitCurve(const std::vector<double>& xs,
                     const std::vector<double>& ys) {
  const int num_xs = xs.size();
  const int num_ys = ys.size();
  assert(num_xs == num_ys);
  int point_num = std::min(num_xs, num_ys);
  assert(point_num >= 4);

  std::vector<double> theta_list;
  length_ = 0;
  s_list_.clear();
  x_list_.clear();
  y_list_.clear();
  s_list_.reserve(point_num);
  x_list_.reserve(point_num);
  y_list_.reserve(point_num);
  theta_list.reserve(point_num);

  s_list_.push_back(0);
  x_list_.push_back(xs.at(0));
  y_list_.push_back(ys.at(0));

  double theta0 = 0;
  double x_pre = xs.at(0);
  double y_pre = ys.at(0);
  for (int i = 1; i < point_num; i++) {
    double x = xs.at(i);
    double y = ys.at(i);
    double local_length = std::hypot(x - x_pre, y - y_pre);
    if (local_length > kEps) {
      length_ += local_length;
      s_list_.push_back(length_);
      x_list_.push_back(x);
      y_list_.push_back(y);
      double theta = atan2(y - y_pre, x - x_pre);
      double delta_theta = theta - theta0;
      if (delta_theta > M_PI) {
        theta -= M_PI * 2;
      } else if (delta_theta < -M_PI) {
        theta += M_PI * 2;
      }
      theta0 = theta;
      theta_list.push_back(theta);

      x_pre = x;
      y_pre = y;
    }
  }
  theta_list.push_back(theta_list.back());
  curve_x_.SetPoints(s_list_, x_list_);
  curve_y_.SetPoints(s_list_, y_list_);
  curve_theta_.SetPoints(s_list_, theta_list);
}

void Curve::FitCurve(const std::vector<double>& xs,
                     const std::vector<double>& ys,
                     std::vector<double>* s_list) {
  const int num_xs = xs.size();
  const int num_ys = ys.size();
  assert(num_xs == num_ys);
  int point_num = std::min(num_xs, num_ys);
  assert(point_num >= 4);

  length_ = 0;
  s_list_.clear();
  x_list_.clear();
  y_list_.clear();
  std::vector<double> theta_list;

  s_list_.reserve(point_num);
  x_list_.reserve(point_num);
  y_list_.reserve(point_num);
  theta_list.reserve(point_num);

  s_list_.push_back(0);
  x_list_.push_back(xs.at(0));
  y_list_.push_back(ys.at(0));

  double theta0 = 0;
  double x_pre = xs.at(0);
  double y_pre = ys.at(0);
  for (int i = 1; i < point_num; i++) {
    double x = xs.at(i);
    double y = ys.at(i);
    double local_length = std::hypot(x - x_pre, y - y_pre);
    if (local_length > kEps) {
      length_ += local_length;
      s_list_.push_back(length_);
      x_list_.push_back(x);
      y_list_.push_back(y);
      double theta = atan2(y - y_pre, x - x_pre);
      double delta_theta = theta - theta0;
      if (delta_theta > M_PI) {
        theta -= M_PI * 2;
      } else if (delta_theta < -M_PI) {
        theta += M_PI * 2;
      }
      theta0 = theta;
      theta_list.push_back(theta);
      x_pre = x;
      y_pre = y;
    }
  }
  theta_list.push_back(theta_list.back());

  curve_x_.SetPoints(s_list_, x_list_);
  curve_y_.SetPoints(s_list_, y_list_);
  curve_theta_.SetPoints(s_list_, theta_list);
  s_list->insert(s_list->end(), s_list_.begin(), s_list_.end());
}

void Curve::GetClosestPointOnCurve(double x0, double y0, bool extention_flag,
                                   double* s, double* dist,
                                   bool set_initial_value) const {
  constexpr int kMaxIteration = 20;
  constexpr double kConvergeThreshold = 1e-3;
  constexpr double kBisectionThreshold = 1.0;

  x0 -= offset_x_;
  y0 -= offset_y_;

  auto [min_s, max_s] = curve_x_.GetRange();
  double s_tmp = 0.0;
  double dist_tmp = 0.0;

  if (!set_initial_value) {
    double mid_1, mid_2, mid_dist_1, mid_dist_2;
    while (fabs(max_s - min_s) > kBisectionThreshold) {
      mid_1 = (max_s - min_s) / 3 + min_s;
      mid_2 = (max_s - min_s) * 2 / 3 + min_s;
      mid_dist_1 = ComputeDist(x0, y0, curve_x_, curve_y_, mid_1);
      mid_dist_2 = ComputeDist(x0, y0, curve_x_, curve_y_, mid_2);
      if (mid_dist_1 > mid_dist_2) {
        min_s = mid_1;
      } else {
        max_s = mid_2;
      }
    }
    s_tmp = min_s;
    dist_tmp = ComputeDist(x0, y0, curve_x_, curve_y_, s_tmp);
  } else {
    s_tmp = *s;
  }

  if (!extention_flag) {
    s_tmp = s_tmp < min_s ? min_s : s_tmp;
    s_tmp = s_tmp > max_s ? max_s : s_tmp;
  }

  double x_s, y_s, x_deriv1_s, y_deriv1_s, x_deriv2_s, y_deriv2_s;
  double d_deriv1, d_deriv2, delta_s;
  double dist_pre = 1e6;

  for (int i = 0; i < kMaxIteration; i++) {
    x_s = curve_x_(s_tmp);
    y_s = curve_y_(s_tmp);
    x_deriv1_s = curve_x_.Deriv1(s_tmp);
    y_deriv1_s = curve_y_.Deriv1(s_tmp);
    x_deriv2_s = curve_x_.Deriv2(s_tmp);
    y_deriv2_s = curve_y_.Deriv2(s_tmp);

    d_deriv1 = 2 * ((x_s - x0) * x_deriv1_s + (y_s - y0) * y_deriv1_s);
    d_deriv2 = 2 * (x_deriv1_s * x_deriv1_s + (x_s - x0) * x_deriv2_s +
                    y_deriv1_s * y_deriv1_s + (y_s - y0) * y_deriv2_s);
    if (std::abs(d_deriv2) < 1e-6) {
      delta_s = d_deriv1;
    } else {
      delta_s = d_deriv1 / d_deriv2;
    }

    dist_tmp = std::hypot(x_s - x0, y_s - y0);
    if (dist_pre - dist_tmp > kConvergeThreshold) {
      if ((s_tmp < min_s || s_tmp > max_s) && !extention_flag) {
        break;
      } else {
        s_tmp = s_tmp - delta_s;
        dist_pre = dist_tmp;
      }
    } else {
      break;
    }
  }

  if (!extention_flag) {
    s_tmp = s_tmp < min_s ? min_s : s_tmp;
    s_tmp = s_tmp > max_s ? max_s : s_tmp;
  }

  *s = s_tmp;
  *dist = dist_tmp;
}

void Curve::GetGlobalClosestPointOnCurve(double x0, double y0,
                                         bool extention_flag, double* s,
                                         double* dist) const {
  constexpr int kMaxIteration = 20;
  constexpr double kConvergeThreshold = 1e-3;
  x0 -= offset_x_;
  y0 -= offset_y_;

  auto [min_s, max_s] = curve_x_.GetRange();

  double closest_dist = 1e6, closest_s = min_s;
  int control_num = s_list_.size();
  for (int i = 0; i < control_num; i++) {
    const double x = x_list_.at(i);
    const double y = y_list_.at(i);
    const double dist = (x - x0) * (x - x0) + (y - y0) * (y - y0);
    if (dist < closest_dist) {
      closest_dist = dist;
      closest_s = s_list_.at(i);
    }
  }

  double x_s, y_s, x_deriv1_s, y_deriv1_s, x_deriv2_s, y_deriv2_s;
  double d_deriv1, d_deriv2, delta_s;
  double dist_pre = 1e6;
  double s_tmp = closest_s;
  double dist_tmp = closest_dist;

  for (int i = 0; i < kMaxIteration; i++) {
    x_s = curve_x_(s_tmp);
    y_s = curve_y_(s_tmp);
    x_deriv1_s = curve_x_.Deriv1(s_tmp);
    y_deriv1_s = curve_y_.Deriv1(s_tmp);
    x_deriv2_s = curve_x_.Deriv2(s_tmp);
    y_deriv2_s = curve_y_.Deriv2(s_tmp);

    d_deriv1 = 2 * ((x_s - x0) * x_deriv1_s + (y_s - y0) * y_deriv1_s);
    d_deriv2 = 2 * (x_deriv1_s * x_deriv1_s + (x_s - x0) * x_deriv2_s +
                    y_deriv1_s * y_deriv1_s + (y_s - y0) * y_deriv2_s);
    if (std::abs(d_deriv2) < 1e-3) {
      delta_s = d_deriv1;
    } else {
      delta_s = d_deriv1 / d_deriv2;
    }

    dist_tmp = std::hypot(x_s - x0, y_s - y0);
    if (dist_pre - dist_tmp > kConvergeThreshold) {
      if ((s_tmp < min_s || s_tmp > max_s) && !extention_flag) {
        break;
      } else {
        s_tmp = s_tmp - delta_s;
        dist_pre = dist_tmp;
      }
    } else {
      break;
    }
  }

  if (!extention_flag) {
    if (s_tmp < min_s || s_tmp > max_s) {
      s_tmp = s_tmp < min_s ? min_s : s_tmp;
      s_tmp = s_tmp > max_s ? max_s : s_tmp;

      x_s = curve_x_(s_tmp);
      y_s = curve_y_(s_tmp);
      dist_tmp = std::hypot(x_s - x0, y_s - y0);
    }
  }

  *s = s_tmp;
  *dist = dist_tmp;
}
}  // namespace geometry
}  // namespace common
}  // namespace roadstar
