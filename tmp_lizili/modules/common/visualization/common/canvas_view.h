#ifndef MODULES_COMMON_VISUALIZATION_COMMON_CANVAS_VIEW_H_
#define MODULES_COMMON_VISUALIZATION_COMMON_CANVAS_VIEW_H_

namespace roadstar {
namespace common {
namespace visualization {

class CanvasView {
 public:
  explicit CanvasView(int w, int h, double view_w, double view_h)
      : w_(w),
        h_(h),
        view_w_(view_w),
        view_h_(view_h),
        scale_x_(w / view_w),
        scale_y_(h / view_h) {}
  CanvasView(const CanvasView&) = default;
  double view_w() const {
    return view_w_;
  }
  double view_h() const {
    return view_h_;
  }
  double utm_x() const {
    return utm_x_;
  }
  double utm_y() const {
    return utm_y_;
  }
  double utm_z() const {
    return utm_z_;
  }
  double heading() const {
    return heading_;
  }
  int w() const {
    return w_;
  }
  int h() const {
    return h_;
  }
  double scale_x() const {
    return scale_x_;
  }
  double scale_y() const {
    return scale_y_;
  }
  void set_view_w(double view_w) {
    view_w_ = view_w;
    scale_x_ = w_ / view_w_;
  }
  void set_view_h(double view_h) {
    view_h_ = view_h;
    scale_y_ = h_ / view_h_;
  }
  void set_utm_x(double utm_x) {
    utm_x_ = utm_x;
  }
  void set_utm_y(double utm_y) {
    utm_y_ = utm_y;
  }
  void set_utm_z(double utm_z) {
    utm_z_ = utm_z;
  }
  void set_heading(double heading) {
    heading_ = heading;
  }

 private:
  const int w_;
  const int h_;
  double view_w_;
  double view_h_;
  double scale_x_;
  double scale_y_;
  double utm_x_ = 0;  // center where the canvas is located on the utm map
  double utm_y_ = 0;
  double utm_z_ = 0;
  double heading_ = 0;  // rotation
};

}  // namespace visualization
}  // namespace common
}  // namespace roadstar

#endif
