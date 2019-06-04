#pragma once

#include <opencv2/opencv.hpp>

#include <cstdint>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "modules/common/proto/colormap.pb.h"
#include "modules/common/util/hash.h"

namespace roadstar::common::util {

class Color {
 public:
  explicit Color(uint8_t r = 0, uint8_t g = 0, uint8_t b = 0, uint8_t a = 0)
      : color_(0) {
    color_ |= (r << 24);
    color_ |= (g << 16);
    color_ |= (b << 8);
    color_ |= a;
  }

  // NOLINTNEXTLINE(google-explicit-constructor)
  Color(const ColorName &name) : color_(map_[name].color_) {}

  inline static Color FromRGB(uint32_t rgb) {
    Color color;
    color.color_ = (rgb << 8);
    return color;
  }

  inline static Color FromRGBA(uint32_t rgba) {
    Color color;
    color.color_ = rgba;
    return color;
  }

  template <typename T = uint8_t>
  inline typename std::enable_if<std::is_integral<T>::value, T>::type r()
      const {
    return static_cast<T>((color_ >> 24) & 0xff);
  }

  template <typename T = uint8_t>
  inline typename std::enable_if<std::is_floating_point<T>::value, T>::type r()
      const {
    return static_cast<T>(r() / 255.0);
  }

  template <typename T = uint8_t>
  inline typename std::enable_if<std::is_integral<T>::value, T>::type g()
      const {
    return static_cast<T>((color_ >> 16) & 0xff);
  }

  template <typename T = uint8_t>
  inline typename std::enable_if<std::is_floating_point<T>::value, T>::type g()
      const {
    return static_cast<T>(g() / 255.0);
  }

  template <typename T = uint8_t>
  inline typename std::enable_if<std::is_integral<T>::value, T>::type b()
      const {
    return static_cast<T>((color_ >> 8) & 0xff);
  }

  template <typename T = uint8_t>
  inline typename std::enable_if<std::is_floating_point<T>::value, T>::type b()
      const {
    return static_cast<T>(b() / 255.0);
  }

  template <typename T = uint8_t>
  inline typename std::enable_if<std::is_integral<T>::value, T>::type a()
      const {
    return static_cast<T>(color_ & 0xff);
  }

  template <typename T = uint8_t>
  inline typename std::enable_if<std::is_floating_point<T>::value, T>::type a()
      const {
    return static_cast<T>(a() / 255.0);
  }

  inline uint32_t rgb() const {
    return color_ >> 8;
  }

  inline uint32_t rgba() const {
    return color_;
  }

  inline cv::Scalar cv_rgb() const {
    return cv::Scalar(b(), g(), r());
  }

 private:
  uint32_t color_;
  static std::unordered_map<ColorName, Color, EnumHash> map_;
};
}  // namespace roadstar::common::util

// NOLINTNEXTLINE(google-runtime-int)
inline roadstar::common::util::Color operator"" _rgb(unsigned long long rgb) {
  return roadstar::common::util::Color::FromRGB(static_cast<uint32_t>(rgb));
}

// NOLINTNEXTLINE(google-runtime-int)
inline roadstar::common::util::Color operator"" _rgba(unsigned long long rgba) {
  return roadstar::common::util::Color::FromRGBA(static_cast<uint32_t>(rgba));
}
