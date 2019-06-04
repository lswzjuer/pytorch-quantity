#ifndef MODULES_COMMON_GEOMETRY_CURVE_H_
#define MODULES_COMMON_GEOMETRY_CURVE_H_

#include <vector>

#include "modules/common/geometry/spline.h"

namespace roadstar {
namespace common {
namespace geometry {
// Spline fit of lane, stores curve x, curve y and curve theta
// and length information
class Curve {
 public:
  double x(double s) const {
    return curve_x_(s) + offset_x_;
  }

  double y(double s) const {
    return curve_y_(s) + offset_y_;
  }

  double theta(double s) const {
    return curve_theta_(s);
  }

  double x_dot(double s) const {
    return curve_x_.Deriv1(s);
  }

  double y_dot(double s) const {
    return curve_y_.Deriv1(s);
  }

  double theta_dot(double s) const {
    return curve_theta_.Deriv1(s);
  }

  double x_ddot(double s) const {
    return curve_x_.Deriv2(s);
  }

  double y_ddot(double s) const {
    return curve_y_.Deriv2(s);
  }

  double theta_ddot(double s) const {
    return curve_theta_.Deriv2(s);
  }

  void SetOffset(double offset_x, double offset_y) {
    offset_x_ = offset_x;
    offset_y_ = offset_y;
  }

  void x(const std::vector<double>& s_list, std::vector<double>* x_list) const {
    curve_x_(s_list, x_list);
    for (auto iter_x = x_list->begin(); iter_x != x_list->end(); iter_x++) {
      *iter_x += offset_x_;
    }
  }

  void y(const std::vector<double>& s_list, std::vector<double>* y_list) const {
    curve_y_(s_list, y_list);
    for (auto iter_y = y_list->begin(); iter_y != y_list->end(); iter_y++) {
      *iter_y += offset_y_;
    }
  }

  void theta(const std::vector<double>& s_list,
             std::vector<double>* theta_list) const {
    curve_theta_(s_list, theta_list);
  }

  void x_dot(const std::vector<double>& s_list,
             std::vector<double>* x_dot_list) const {
    curve_x_.Deriv1(s_list, x_dot_list);
  }

  void y_dot(const std::vector<double>& s_list,
             std::vector<double>* y_dot_list) const {
    curve_y_.Deriv1(s_list, y_dot_list);
  }

  void theta_dot(const std::vector<double>& s_list,
                 std::vector<double>* theta_dot_list) const {
    curve_theta_.Deriv1(s_list, theta_dot_list);
  }

  void x_ddot(const std::vector<double>& s_list,
              std::vector<double>* x_ddot_list) const {
    curve_x_.Deriv2(s_list, x_ddot_list);
  }

  void y_ddot(const std::vector<double>& s_list,
              std::vector<double>* y_ddot_list) const {
    curve_y_.Deriv2(s_list, y_ddot_list);
  }

  void theta_ddot(const std::vector<double>& s_list,
                  std::vector<double>* theta_ddot_list) const {
    curve_theta_.Deriv2(s_list, theta_ddot_list);
  }

  void InterpolateX(const std::vector<double>& s_list,
                    std::vector<double>* x_list,
                    std::vector<double>* x_dot_list,
                    std::vector<double>* x_ddot_list) const {
    curve_x_.Interpolate(s_list, x_list, x_dot_list, x_ddot_list);
    for (auto iter_x = x_list->begin(); iter_x != x_list->end(); iter_x++) {
      *iter_x += offset_x_;
    }
  }

  void InterpolateY(const std::vector<double>& s_list,
                    std::vector<double>* y_list,
                    std::vector<double>* y_dot_list,
                    std::vector<double>* y_ddot_list) const {
    curve_y_.Interpolate(s_list, y_list, y_dot_list, y_ddot_list);
    for (auto iter_y = y_list->begin(); iter_y != y_list->end(); iter_y++) {
      *iter_y += offset_y_;
    }
  }

  void InterpolateTheta(const std::vector<double>& s_list,
                        std::vector<double>* theta_list,
                        std::vector<double>* theta_dot_list,
                        std::vector<double>* theta_ddot_list) const {
    curve_theta_.Interpolate(s_list, theta_list, theta_dot_list,
                             theta_ddot_list);
  }

  double length() const {
    return length_;
  }

  void FitCurve(const std::vector<double>& xs, const std::vector<double>& ys);

  void FitCurve(const std::vector<double>& xs, const std::vector<double>& ys,
                std::vector<double>* s_list);

  void GetClosestPointOnCurve(double x0, double y0, bool extension_flag,
                              double* s, double* dist,
                              bool set_initial_value = false) const;

  void GetGlobalClosestPointOnCurve(double x0, double y0, bool extension_flag,
                                    double* s, double* dist) const;

  void Clear();

  int control_point_num() const {
    return s_list_.size();
  }

  double control_point_x(int index) const {
    if (index < static_cast<int>(x_list_.size())) {
      return x_list_.at(index) + offset_x_;
    }
    return 0;
  }

  double control_point_y(int index) const {
    if (index < static_cast<int>(y_list_.size())) {
      return y_list_.at(index) + offset_y_;
    }
    return 0;
  }

  double control_point_s(int index) const {
    if (index < static_cast<int>(s_list_.size())) {
      return s_list_.at(index);
    }
    return 0;
  }

 private:
  Spline curve_x_;
  Spline curve_y_;
  Spline curve_theta_;

  std::vector<double> s_list_, x_list_, y_list_;

  double length_ = 0;

  double offset_x_ = 0, offset_y_ = 0;

  constexpr static double kEps = 1e-3;
};
}  // namespace geometry
}  // namespace common
}  // namespace roadstar

#endif  // MODULES_COMMON_GEOMETRY_CURVE_H_
