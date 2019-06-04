#ifndef MODULES_COMMON_GEOMETRY_SPLINE_H_
#define MODULES_COMMON_GEOMETRY_SPLINE_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <tuple>
#include <vector>

namespace roadstar {
namespace common {
namespace geometry {
// Band matrix
// | d u u         |
// | l d u u       |
// | l l d u u     |
// |   l l d u u   |
// |     l l d u u |
// |       l l d u |
// |         l l d |
// Band matrix, the image above describes a BandMatrix
// u is the upper, l is the lower, d is the diag
class BandMatrix {
 public:
  BandMatrix(int dim, int num_upper, int num_lower) {
    lu_decomposed_ = false;
    Resize(dim, num_upper, num_lower);
  }

  // Matrix dimension
  int Dim() const {
    return diag_.size();
  }

  int NumUpper() const {
    return upper_.size();
  }

  int NumLower() const {
    return lower_.size();
  }

  // Access operator
  void SetValue(int i, int j, double value);
  // Defines the new operator (), so that we can access the elements
  // by A(i,j), index going from i=0,..., dim()-1
  double operator()(int i, int j) const;

  // we can store an additional diogonal
  void SetSavedDiag(int i, double value);
  double SavedDiag(int i) const;

  // LUDecomposition of a band matrix, reference:
  // https://en.wikipedia.org/wiki/LU_decomposition
  void LUDecompose();

  // Solves Ux = y
  std::vector<double> RightSolve(const std::vector<double>& b) const;
  // Solves Ly = b
  std::vector<double> LeftSolve(const std::vector<double>& b) const;
  std::vector<double> LUSolve(const std::vector<double>& b);

 private:
  // Diagonal line
  std::vector<double> diag_;

  // Save additional diagonal information for LUDecompose computation
  std::vector<double> saved_diag_;

  // Upper band
  std::vector<std::vector<double>> upper_;

  // Lower band
  std::vector<std::vector<double>> lower_;

  bool lu_decomposed_;

  // Resize the matrix
  void Resize(int dim, int num_upper, int num_lower);
};

// Spline interpolation, Reference:
// https://en.wikiversity.org/wiki/Cubic_Spline_Interpolation
class Spline {
 public:
  enum class BandType { kFirstDeriv = 1, kSecondDeriv = 2 };

  // Optional, but if called it has to come be before setPoints()
  void SetBoundary(BandType left, double left_value, BandType right,
                   double right_value, bool force_linear_extrapolation = false);

  void SetPoints(const std::vector<double>& x, const std::vector<double>& y,
                 bool cubic_Spline = true);

  // Compute the interpolation value f(x) given x
  double operator()(double x) const;

  void operator()(const std::vector<double>& xs, std::vector<double>* ys) const;

  // Compute the first derivative f'(x) given x
  double Deriv1(double x) const;

  void Deriv1(const std::vector<double>& xs, std::vector<double>* ys) const;

  // Compute the second derivative f''(x) given x
  double Deriv2(double x) const;

  void Deriv2(const std::vector<double>& xs, std::vector<double>* ys) const;

  void Interpolate(const std::vector<double>& xs, std::vector<double>* ys,
                   std::vector<double>* y_dots,
                   std::vector<double>* y_ddots) const;
  // Return the range of x
  // Output:
  //      return <min_value, max_value> of x
  std::tuple<double, double> GetRange() const;

  void Clear();

 private:
  // x,y coordinates of points
  std::vector<double> x_;
  std::vector<double> y_;

  // Interpolation parameters, Spline coefficients
  // f(x) = a_i * (x - x_i)^3 + b_i * (x - x_i)^2 + c_i * (x - x_i) + y_i
  std::vector<double> a_;
  std::vector<double> b_;
  std::vector<double> c_;

  // For left extrapol
  double b0_ = 0.0;
  double c0_ = 0.0;

  BandType left_ = BandType::kSecondDeriv;
  BandType right_ = BandType::kSecondDeriv;

  double left_value_ = 0.0;
  double right_value_ = 0.0;
  bool force_linear_extrapolation_ = false;
};
}  // namespace geometry
}  // namespace common
}  // namespace roadstar

#endif  // MODULES_COMMON_GEOMETRY_SPLINE_H_
