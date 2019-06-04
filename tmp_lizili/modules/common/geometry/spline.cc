#include "modules/common/geometry/spline.h"

#include <assert.h>
#include <algorithm>
#include <iostream>
#include <tuple>
#include <vector>

namespace roadstar {
namespace common {
namespace geometry {

void BandMatrix::Resize(int dim, int num_upper, int num_lower) {
  diag_.resize(dim);
  saved_diag_.resize(dim);

  upper_.resize(num_upper);
  lower_.resize(num_lower);

  for (size_t i = 0; i < upper_.size(); i++) {
    upper_[i].resize(dim);
  }

  for (size_t i = 0; i < lower_.size(); i++) {
    lower_[i].resize(dim);
  }
}

void BandMatrix::SetValue(int i, int j, double value) {
  const int k = j - i;  // what band is the entry

  assert((i >= 0) && (i < Dim()) && (j >= 0) && (j < Dim()));
  assert((-NumLower() <= k) && (k <= NumUpper()));

  if (k == 0) {
    // If k = 0 -> diogonal.
    diag_[i] = value;
  } else if (k > 0) {
    // If k > 0 upper right part.
    upper_[k - 1][i] = value;
  } else {
    // If k < 0 lower left part.
    lower_[-k - 1][i] = value;
  }
}

double BandMatrix::operator()(int i, int j) const {
  const int k = j - i;  // what band is the entry

  assert((i >= 0) && (i < Dim()) && (j >= 0) && (j < Dim()));
  assert((-NumLower() <= k) && (k <= NumUpper()));

  if (k == 0) {
    // If k = 0 -> diogonal.
    return diag_[i];
  } else if (k > 0) {
    // If k > 0 upper right part.
    return upper_[k - 1][i];
  } else {
    // If k < 0 lower left part.
    return lower_[-k - 1][i];
  }
}

double BandMatrix::SavedDiag(int i) const {
  assert((i >= 0) && (i < Dim()));
  return saved_diag_[i];
}

void BandMatrix::SetSavedDiag(int i, double value) {
  assert((i >= 0) && (i < Dim()));
  saved_diag_[i] = value;
}

void BandMatrix::LUDecompose() {
  // Preconditioning
  // Normalize column i so that L(i, i) = 1
  for (int i = 0; i < Dim(); i++) {
    // saved_diag_(i) = 1 / diag_(i)
    SetSavedDiag(i, 1.0 / (*this)(i, i));

    // Compute the min value and max value of j
    const int j_min = std::max(0, i - NumLower());
    const int j_max = std::min(Dim() - 1, i + NumUpper());

    for (int j = j_min; j <= j_max; j++) {
      // A(i, j) = A(i, j) / A(i, i)
      SetValue(i, j, (*this)(i, j) * SavedDiag(i));
    }

    // Prevents rounding errors
    SetValue(i, i, 1.0);
  }

  // Gauss LU-Decomposition
  for (int k = 0; k < Dim(); k++) {
    const int i_max = std::min(Dim() - 1, k + NumLower());

    for (int i = k + 1; i <= i_max; i++) {
      const double x = -(*this)(i, k) / (*this)(k, k);

      // Assembly part of L
      // L(i, k) = A(i, k) / A(k, k)
      SetValue(i, k, -x);
      const int j_max = std::min(Dim() - 1, k + NumUpper());

      for (int j = k + 1; j <= j_max; j++) {
        // Assembly part of R
        // R(i, j) = A(i, j) + \sum_k A(i, k) * A(k, j) / A(k, k)
        SetValue(i, j, (*this)(i, j) + x * (*this)(k, j));
      }
    }
  }

  lu_decomposed_ = true;
}

std::vector<double> BandMatrix::LeftSolve(const std::vector<double>& b) const {
  std::vector<double> x(Dim());

  for (int i = 0; i < Dim(); i++) {
    double sum = 0.0;
    const int j_start = std::max(0, i - NumLower());

    for (int j = j_start; j < i; j++) {
      sum += (*this)(i, j) * x[j];
    }

    x[i] = (b[i] * SavedDiag(i)) - sum;
  }

  return x;
}

std::vector<double> BandMatrix::RightSolve(const std::vector<double>& b) const {
  std::vector<double> x(this->Dim());

  for (int i = Dim() - 1; i >= 0; i--) {
    double sum = 0.0;
    const int j_stop = std::min(Dim() - 1, i + NumUpper());
    for (int j = i + 1; j <= j_stop; j++) {
      sum += (*this)(i, j) * x[j];
    }
    x[i] = (b[i] - sum) / (*this)(i, i);
  }
  return x;
}

std::vector<double> BandMatrix::LUSolve(const std::vector<double>& b) {
  std::vector<double> y;

  if (!lu_decomposed_) {
    LUDecompose();
  }

  y = LeftSolve(b);
  return RightSolve(y);
}

void Spline::SetBoundary(Spline::BandType left, double left_value,
                         Spline::BandType right, double right_value,
                         bool force_linear_extrapolation) {
  // SetPoints() must not have happened yet
  left_ = left;
  right_ = right;
  left_value_ = left_value;
  right_value_ = right_value;
  force_linear_extrapolation_ = force_linear_extrapolation;
}

void Spline::SetPoints(const std::vector<double>& x,
                       const std::vector<double>& y, bool cubic_spline) {
  x_ = x;
  y_ = y;
  const int n = x.size();

  for (int i = 0; i < n - 1; i++) {
    assert(x_[i] < x_[i + 1]);
  }

  // Cubic Spline interpolation
  if (cubic_spline) {
    // Setting up the matrix and right hand side of the equation system
    // for the parameters b[]
    BandMatrix A(n, 1, 1);
    std::vector<double> rhs(n);

    // The matrix A
    // | 2.0                                                             |
    // | (x(i) - x(i-1))/3  2 * (x(i+1) - x(i-1))/3   (x(i+1) - x(i))/3  |
    // |                                              2.0                |
    // The right hand vector rhs
    // [left_value,
    // (y(i+1)-y(i))/(x(i+1)-x(i)) - (y(i)-y(i-1))/(x(i)-x(i-1)),
    // ... ,
    // right_value]
    for (int i = 1; i < n - 1; i++) {
      A.SetValue(i, i - 1, 1.0 / 3.0 * (x[i] - x[i - 1]));
      A.SetValue(i, i, 2.0 / 3.0 * (x[i + 1] - x[i - 1]));
      A.SetValue(i, i + 1, 1.0 / 3.0 * (x[i + 1] - x[i]));
      rhs[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) -
               (y[i] - y[i - 1]) / (x[i] - x[i - 1]);
    }

    // Boundary conditions
    if (left_ == Spline::BandType::kSecondDeriv) {
      // 2 * b[0] = f''
      A.SetValue(0, 0, 2.0);
      A.SetValue(0, 1, 0.0);
      rhs[0] = left_value_;
    } else if (left_ == Spline::BandType::kFirstDeriv) {
      // c[0] = f', needs to be re-expressed in terms of b:
      // (2b[0]+b[1])(x[1]-x[0]) = 3 ((y[1]-y[0])/(x[1]-x[0]) - f')
      A.SetValue(0, 0, 2.0 * (x[1] - x[0]));
      A.SetValue(0, 1, 1.0 * (x[1] - x[0]));
      rhs[0] = 3.0 * ((y[1] - y[0]) / (x[1] - x[0]) - left_value_);
    }

    if (right_ == Spline::BandType::kSecondDeriv) {
      // 2 * b[n-1] = f''
      A.SetValue(n - 1, n - 1, 2.0);
      A.SetValue(n - 1, n - 2, 0.0);
      rhs[n - 1] = right_value_;
    } else if (right_ == Spline::BandType::kFirstDeriv) {
      // c[n - 1] = f', needs to be re-expressed in terms of b:
      // (b[n-2] + 2b[n-1])(x[n-1] - x[n-2])
      // = 3 (f' - (y[n-1]-y[n-2])/(x[n-1]-x[n-2]))
      A.SetValue(n - 1, n - 1, 2.0 * (x[n - 1] - x[n - 2]));
      A.SetValue(n - 1, n - 2, 1.0 * (x[n - 1] - x[n - 2]));
      rhs[n - 1] =
          3.0 * (right_value_ - (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]));
    }
    // Solve the equation system to obtain the parameters b[]
    // A b_ = rhs
    b_ = A.LUSolve(rhs);

    // Calculate parameters a[] and c[] based on b[]
    a_.resize(n);
    c_.resize(n);
    for (int i = 0; i < n - 1; i++) {
      // a_i = \frac{b(i+1) - b(i)} {3*(x(i+1) - x(i))}
      a_[i] = 1.0 / 3.0 * (b_[i + 1] - b_[i]) / (x[i + 1] - x[i]);

      // c_i = \frac{y_{i+1} - y_i} {x_{i+1} - x_i}
      //       - \frac{2 b_i + b_{i+1}*(x_{i+1} - x_i)} {3}
      c_[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) -
              1.0 / 3.0 * (2.0 * b_[i] + b_[i + 1]) * (x[i + 1] - x[i]);
    }
  } else {
    // Linear interpolation
    // a_i = 0.0
    // b_i = 0.0
    // c_i = \frac{y_{i+1} - y_i} {x_{i+1}, x_i}
    a_.resize(n);
    b_.resize(n);
    c_.resize(n);

    for (int i = 0; i < n - 1; i++) {
      a_[i] = 0.0;
      b_[i] = 0.0;
      c_[i] = (y_[i + 1] - y_[i]) / (x_[i + 1] - x_[i]);
    }
  }

  // Compute the left and the right boundary
  // For left extrapolation coefficients
  b0_ = force_linear_extrapolation_ ? 0.0 : b_[0];
  c0_ = c_[0];

  // for the right extrapolation coefficients
  // f_{n-1}(x) = b*(x-x_{n-1})^2 + c*(x-x_{n-1}) + y_{n-1}
  const double h = x[n - 1] - x[n - 2];

  // b_[n-1] is determined by the boundary condition
  a_[n - 1] = 0.0;

  // = f'_{n-2}(x_{n-1})
  c_[n - 1] = 3.0 * a_[n - 2] * h * h + 2.0 * b_[n - 2] * h + c_[n - 2];

  if (force_linear_extrapolation_) {
    b_[n - 1] = 0.0;
  }
}

double Spline::operator()(double x) const {
  size_t n = x_.size();

  // 1. Find the closest point x_[idx] < x,
  // idx = 0 even if x < x_[0]
  std::vector<double>::const_iterator it;
  it = std::lower_bound(x_.begin(), x_.end(), x);
  const int idx = std::max(static_cast<int>(it - x_.begin()) - 1, 0);

  const double h = x - x_[idx];
  double interpol = 0.0;
  if (x < x_[0]) {
    // Extrapolation to the left
    // f(x) = b_0 * (x - x_0)^2 + c_0 * (x - x_0) + y_0
    interpol = (b0_ * h + c0_) * h + y_[0];
  } else if (x > x_[n - 1]) {
    // Extrapolation to the right
    // f(x) = b_{n-1} * (x - x_{n-1})^2 + c_{n-1} * (x - x_0) + y_0
    interpol = (b_[n - 1] * h + c_[n - 1]) * h + y_[n - 1];
  } else {
    // Interpolation
    // f(x) = a_i * (x - x_i)^3 + b_i * (x - x_i)^2 + c_i * (x - x_i) + y_i
    interpol = ((a_[idx] * h + b_[idx]) * h + c_[idx]) * h + y_[idx];
  }

  return interpol;
}

void Spline::operator()(const std::vector<double>& xs,
                        std::vector<double>* ys) const {
  if (ys->size() != xs.size()) {
    ys->resize(xs.size());
  }
  size_t n = x_.size();
  std::vector<double>::const_iterator it = x_.begin();
  for (auto iter_x = xs.begin(); iter_x < xs.end() - 1; iter_x++) {
    assert(*iter_x <= *(iter_x + 1));
  }
  double x = 0;
  int index = 0;
  for (auto iter_x = xs.begin(); iter_x < xs.end(); iter_x++, index++) {
    x = *iter_x;
    it = std::lower_bound(it, x_.end(), x);
    const int idx = std::max(static_cast<int>(it - x_.begin()) - 1, 0);

    const double h = x - x_[idx];
    double interpol = 0.0;
    if (x < x_[0]) {
      interpol = (b0_ * h + c0_) * h + y_[0];
    } else if (x > x_[n - 1]) {
      interpol = (b_[n - 1] * h + c_[n - 1]) * h + y_[n - 1];
    } else {
      interpol = ((a_[idx] * h + b_[idx]) * h + c_[idx]) * h + y_[idx];
    }
    ys->at(index) = interpol;
  }
}

double Spline::Deriv1(double x) const {
  const size_t n = x_.size();

  // Find the closest point x_[idx] < x
  // idx = 0 even if x < x_[0]
  std::vector<double>::const_iterator it;
  it = std::lower_bound(x_.begin(), x_.end(), x);
  const int idx = std::max(static_cast<int>(it - x_.begin()) - 1, 0);

  const double h = x - x_[idx];
  double interpol = 0.0;
  if (x < x_[0]) {
    // Extrapolation to the left
    // f'(x) = 2 b_0 (x - x_0) + c_0
    interpol = 2 * b0_ * h + c0_;
  } else if (x > x_[n - 1]) {
    // Extrapolation to the right
    // f'(x) = 2 b_{n-1} (x - x_{n-1}) + c_{n-1}
    interpol = 2 * b_[n - 1] * h + c_[n - 1];
  } else {
    // Interpolation
    // f'(x) = 3 a_i * (x - x_i)^2 + 2 b_i (x - x_i) + c_i
    interpol = (3 * a_[idx] * h + 2 * b_[idx]) * h + c_[idx];
  }

  return interpol;
}

void Spline::Deriv1(const std::vector<double>& xs,
                    std::vector<double>* ys) const {
  if (ys->size() != xs.size()) {
    ys->resize(xs.size());
  }
  size_t n = x_.size();
  std::vector<double>::const_iterator it = x_.begin();
  for (auto iter_x = xs.begin(); iter_x < xs.end() - 1; iter_x++) {
    assert(*iter_x <= *(iter_x + 1));
  }
  double x = 0;
  int index = 0;
  for (auto iter_x = xs.begin(); iter_x < xs.end(); iter_x++, index++) {
    x = *iter_x;
    it = std::lower_bound(it, x_.end(), x);
    const int idx = std::max(static_cast<int>(it - x_.begin()) - 1, 0);

    const double h = x - x_[idx];
    double interpol = 0.0;
    if (x < x_[0]) {
      interpol = 2 * b0_ * h + c0_;
    } else if (x > x_[n - 1]) {
      interpol = 2 * b_[n - 1] * h + c_[n - 1];
    } else {
      interpol = (3 * a_[idx] * h + 2 * b_[idx]) * h + c_[idx];
    }
    ys->at(index) = interpol;
  }
}

double Spline::Deriv2(double x) const {
  const size_t n = x_.size();

  // Find the closest point x_[idx] < x
  // idx = 0 even if x < x_[0]
  std::vector<double>::const_iterator it;
  it = std::lower_bound(x_.begin(), x_.end(), x);
  const int idx = std::max(static_cast<int>(it - x_.begin()) - 1, 0);

  const double h = x - x_[idx];
  double interpol = 0.0;
  if (x < x_[0]) {
    // Extrapolation to the left
    // f''(x) = 2 b_0
    interpol = 2 * b0_;
  } else if (x > x_[n - 1]) {
    // Extrapolation to the right
    // f''(x) = 2 b_{n-1}
    interpol = 2 * b_[n - 1];
  } else {
    // Interpolation
    // f''(x) = 6 a_i (x - x_i) + 2 b_i
    interpol = 6 * a_[idx] * h + 2 * b_[idx];
  }

  return interpol;
}

void Spline::Deriv2(const std::vector<double>& xs,
                    std::vector<double>* ys) const {
  if (ys->size() != xs.size()) {
    ys->resize(xs.size());
  }
  size_t n = x_.size();
  std::vector<double>::const_iterator it = x_.begin();
  for (auto iter_x = xs.begin(); iter_x < xs.end() - 1; iter_x++) {
    assert(*iter_x <= *(iter_x + 1));
  }
  double x = 0;
  int index = 0;
  for (auto iter_x = xs.begin(); iter_x < xs.end(); iter_x++, index++) {
    x = *iter_x;
    it = std::lower_bound(it, x_.end(), x);
    const int idx = std::max(static_cast<int>(it - x_.begin()) - 1, 0);

    const double h = x - x_[idx];
    double interpol = 0.0;
    if (x < x_[0]) {
      interpol = 2 * b0_;
    } else if (x > x_[n - 1]) {
      interpol = 2 * b_[n - 1];
    } else {
      interpol = 6 * a_[idx] * h + 2 * b_[idx];
    }
    ys->at(index) = interpol;
  }
}

void Spline::Interpolate(const std::vector<double>& xs, std::vector<double>* ys,
                         std::vector<double>* y_dots,
                         std::vector<double>* y_ddots) const {
  if (ys->size() != xs.size()) {
    ys->resize(xs.size());
  }
  if (y_dots->size() != xs.size()) {
    y_dots->resize(xs.size());
  }
  if (y_ddots->size() != xs.size()) {
    y_ddots->resize(xs.size());
  }

  size_t n = x_.size();
  std::vector<double>::const_iterator it = x_.begin();
  for (auto iter_x = xs.begin(); iter_x < xs.end() - 1; iter_x++) {
    assert(*iter_x <= *(iter_x + 1));
  }
  double x = 0;
  int index = 0;
  for (auto iter_x = xs.begin(); iter_x < xs.end(); iter_x++, index++) {
    x = *iter_x;
    it = std::lower_bound(it, x_.end(), x);
    const int idx = std::max(static_cast<int>(it - x_.begin()) - 1, 0);

    const double h = x - x_[idx];
    double interpol = 0.0, interpol_dot = 0.0, interpol_ddot = 0.0;
    if (x < x_[0]) {
      interpol = (b0_ * h + c0_) * h + y_[0];
      interpol_dot = 2 * b0_ * h + c0_;
      interpol_ddot = 2 * b0_;
    } else if (x > x_[n - 1]) {
      interpol = (b_[n - 1] * h + c_[n - 1]) * h + y_[n - 1];
      interpol_dot = 2 * b_[n - 1] * h + c_[n - 1];
      interpol_ddot = 2 * b_[n - 1];
    } else {
      interpol = ((a_[idx] * h + b_[idx]) * h + c_[idx]) * h + y_[idx];
      interpol_dot = (3 * a_[idx] * h + 2 * b_[idx]) * h + c_[idx];
      interpol_ddot = 6 * a_[idx] * h + 2 * b_[idx];
    }

    ys->at(index) = interpol;
    y_dots->at(index) = interpol_dot;
    y_ddots->at(index) = interpol_ddot;
  }
}

std::tuple<double, double> Spline::GetRange() const {
  return std::make_tuple(x_.front(), x_.back());
}

void Spline::Clear() {
  x_.clear();
  y_.clear();
  a_.clear();
  b_.clear();
  c_.clear();
}
}  // namespace geometry
}  // namespace common
}  // namespace roadstar
