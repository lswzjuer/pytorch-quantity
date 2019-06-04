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

#ifndef MODULES_COMMON_VISUALIZATION_REMOTE_VISUALIZER_REMOTE_VISUALIZER_H_
#define MODULES_COMMON_VISUALIZATION_REMOTE_VISUALIZER_REMOTE_VISUALIZER_H_

#include <cstdint>
#include <deque>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <utility>
#include <vector>
#include "grpc++/grpc++.h"
#include "modules/common/log.h"
#include "modules/common/macro.h"
#include "modules/common/proto/geometry.pb.h"
#include "modules/common/visualization/remote_visualizer/proto/remote_visualizer.grpc.pb.h"
#include "modules/common/visualization/remote_visualizer/proto/remote_visualizer.pb.h"
#include "modules/common/visualization/remote_visualizer/proto/remote_visualizer_grpc.grpc.pb.h"
#include "modules/common/visualization/remote_visualizer/proto/remote_visualizer_grpc.pb.h"
#include "modules/common/visualization/remote_visualizer/remote_visualizer_gflags.h"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"

namespace roadstar::common::visualization {

class BaseWindowHandler;
template <WindowType::Type Type>
class WindowHandler;
class BaseObjectHandler {
 public:
  BaseObjectHandler(uint32_t id, std::shared_ptr<BaseWindowHandler> parent)
      : id_(id), parent_(parent) {
    color_.set_red(255);
    color_.set_blue(255);
    color_.set_green(255);
  }
  virtual ~BaseObjectHandler();

 protected:
  uint32_t id_;
  Color color_;
  std::shared_ptr<BaseWindowHandler> parent_;
  void SetProporty(const Color &color, bool is_update = true);
  inline uint32_t GetParentId() const;
  virtual bool Update(bool is_update) = 0;
  virtual bool Update() {
    return Update(true);
  }
};

class BaseObject3DHandler : public BaseObjectHandler {
 public:
  using Translation = Vector3D;
  BaseObject3DHandler(uint32_t id, std::shared_ptr<BaseWindowHandler> parent)
      : BaseObjectHandler(id, parent) {
    SetProporty(1, false);
    Translation translation;
    translation.set_x(0);
    translation.set_y(0);
    translation.set_z(0);
    SetProporty(translation, false);
    Rotation rotation;
    rotation.set_angle(0);
    Vector3D *rotation_axis = rotation.mutable_vector();
    rotation_axis->set_x(0);
    rotation_axis->set_y(0);
    rotation_axis->set_z(0);
    SetProporty(rotation, false);
  }
  using BaseObjectHandler::SetProporty;
  void SetProporty(float scale, bool is_update = true);
  void SetProporty(const Translation &translation, bool is_update = true);
  void SetProporty(const Rotation &rotation, bool is_update = true);

 protected:
  Transform transform_;
  bool Update(bool is_update);
  inline void SetObjectInfo(Object3D *object) const;

 private:
  friend class RemoteVisualizer;
  virtual Object3D GetObject3D() const = 0;
};

template <Object3DType::Type Type>
class Object3DHandler : public std::false_type {
  static_assert(std::is_base_of_v<std::false_type, Object3DHandler<Type>>,
                "Unsupported Object3DType");
};

template <>
class Object3DHandler<Object3DType::Pointcloud> : public BaseObject3DHandler {
 public:
  using BaseObject3DHandler::BaseObject3DHandler;
  void SetPointCloud(roadstar::drivers::lidar::PointCloud point_cloud,
                     bool is_update = true);
  Object3D GetObject3D() const override;

 private:
  roadstar::drivers::lidar::PointCloud point_cloud_;
  friend class WindowHandler<WindowType::Window3D>;
  template <typename... Args>
  void SetData(roadstar::drivers::lidar::PointCloud point_cloud,
               Args &&... args);
};

template <>
class Object3DHandler<Object3DType::Plane> : public BaseObject3DHandler {
 public:
  using BaseObject3DHandler::BaseObject3DHandler;
  void SetPlane(float width, float height, bool is_update = true);
  Object3D GetObject3D() const override;

 private:
  Plane plane_;
  friend class WindowHandler<WindowType::Window3D>;
  template <typename... Args>
  void SetData(float width, float height, Args &&... args);
};

template <>
class Object3DHandler<Object3DType::Points> : public BaseObject3DHandler {
 public:
  using BaseObject3DHandler::BaseObject3DHandler;

  void SetPoints(const Points &points, bool is_update = true);
  bool SetRenderType(Object3DType::Type, bool is_update = true);

  Object3D GetObject3D() const override;

 private:
  Object3DType::Type type_;
  Points points_;
  friend class WindowHandler<WindowType::Window3D>;
  template <typename... Args>
  void SetData(const Points &points, Args &&... args,
               Object3DType::Type render_type = Object3DType::Points);
};

template <>
class Object3DHandler<Object3DType::Sphere> : public BaseObject3DHandler {
 public:
  using BaseObject3DHandler::BaseObject3DHandler;
  void SetSphere(int rings, int slices, float radius, bool is_update = true);
  Object3D GetObject3D() const override;

 private:
  Sphere sphere_;
  friend class WindowHandler<WindowType::Window3D>;
  template <typename... Args>
  void SetData(int rings, int slices, float radius, Args &&... args);
};
template <>
class Object3DHandler<Object3DType::Text> : public BaseObject3DHandler {
 public:
  using BaseObject3DHandler::BaseObject3DHandler;
  void SetText(std::string context, float depth, bool is_update = true);
  void SetText(const Text &text, bool is_update = true);

  Object3D GetObject3D() const override;

 private:
  Text text_;
  friend class WindowHandler<WindowType::Window3D>;
  template <typename... Args>
  void SetData(std::string context, float depth, Args &&... args);
};
class BaseObject2DHandler : public BaseObjectHandler {
 public:
  using Translation = roadstar::common::Point2D;
  using BaseObjectHandler::SetProporty;

  BaseObject2DHandler(uint32_t id, std::shared_ptr<BaseWindowHandler> parent)
      : BaseObjectHandler(id, parent) {
    SetProporty(1, false);
    Translation translation;
    translation.set_x(0);
    translation.set_y(0);
    SetProporty(translation, false);
    Rotation2D rotation;
    rotation.set_angle(0);
    Translation *rotation_center = rotation.mutable_center();
    rotation_center->set_x(0);
    rotation_center->set_y(0);
    SetProporty(rotation, false);
  }

  void SetProporty(double scale, bool is_update = true);
  void SetProporty(const Translation &translation, bool is_update = true);
  void SetProporty(const Rotation2D &rotation, bool is_update = true);

 protected:
  inline virtual void SetObjectInfo(Object2D *object) const;
  bool Update(bool is_update);

 private:
  Transform2D transform_;
  friend class RemoteVisualizer;
  virtual Object2D GetObject2D() const = 0;
};

class BasePointsObject2DHandler : public BaseObject2DHandler {
 public:
  using BaseObject2DHandler::BaseObject2DHandler;
  void AddPoint(double x, double y, bool is_update = true);
  inline void AddPoints(const std::vector<Point2D> &points);
  void SetPoints(std::vector<Point2D> points, bool is_update = true);
  void SetSize(double size, bool is_update = true) {
    size_ = size;
    Update(is_update);
  }
  void ClearPoints();

 protected:
  std::vector<roadstar::common::Point2D> points_;
  double size_ = 1;
  inline void SetObjectInfo(Object2D *object) const override;
  friend class WindowHandler<WindowType::Window2D>;
  template <typename... Args>
  void SetData(std::vector<Point2D> points, int width, Args &&... args);
};
template <Object2DType::Type Type>
class Object2DHandler : public std::false_type {
  static_assert(std::is_base_of_v<std::false_type, Object2DHandler<Type>>,
                "Unsupported Object3DType");
};

template <>
class Object2DHandler<Object2DType::Points> : public BasePointsObject2DHandler {
 public:
  using BasePointsObject2DHandler::BasePointsObject2DHandler;

  Object2D GetObject2D() const override;
};
template <>
class Object2DHandler<Object2DType::Line> : public BasePointsObject2DHandler {
 public:
  using BasePointsObject2DHandler::BasePointsObject2DHandler;

  Object2D GetObject2D() const override;
};
template <>
class Object2DHandler<Object2DType::Polygon>
    : public BasePointsObject2DHandler {
 public:
  using BasePointsObject2DHandler::BasePointsObject2DHandler;

  Object2D GetObject2D() const override;
};
template <>
class Object2DHandler<Object2DType::Image> : public BaseObject2DHandler {
 public:
  using BaseObject2DHandler::BaseObject2DHandler;
  bool SetImage(cv::Mat mat, bool is_update = true);
  inline void SetObjectInfo(Object2D *object) const override;

  Object2D GetObject2D() const override;

 private:
  double basic_scale_ = 1;
  friend class WindowHandler<WindowType::Window2D>;
  template <typename... Args>
  void SetData(cv::Mat mat, Args &&... args);
  roadstar::drivers::Image image_;
};
template <>
class Object2DHandler<Object2DType::Chart> : public BaseObject2DHandler {
 public:
  using BaseObject2DHandler::BaseObject2DHandler;
  void SetChart(const Chart &chart, bool is_update = true);
  void SetChartType(ChartType::Type type, bool is_update = true);
  Object2D GetObject2D() const override;

 private:
  friend class WindowHandler<WindowType::Window2D>;
  template <typename... Args>
  void SetData(Chart chart, Args &&... args);
  Chart chart_;
};
template <>
class Object2DHandler<Object2DType::Text> : public BaseObject2DHandler {
 public:
  using BaseObject2DHandler::BaseObject2DHandler;
  void SetText(const std::string &text, bool is_update = true);
  Object2D GetObject2D() const override;

 private:
  friend class WindowHandler<WindowType::Window2D>;
  template <typename... Args>
  void SetData(std::string text, Args &&... args);
  std::string text_;
};
// Window
class BaseWindowHandler
    : public std::enable_shared_from_this<BaseWindowHandler> {
 public:
  explicit BaseWindowHandler(uint32_t id) : id_(id) {}
  inline auto GetId() const {
    return id_;
  }
  virtual ~BaseWindowHandler();

 protected:
  uint32_t id_;
  std::vector<std::unique_ptr<BaseObjectHandler>> objects_;
};
template <WindowType::Type Type>
class WindowHandler : public std::false_type {
  static_assert(std::is_base_of_v<std::false_type, WindowHandler<Type>>, "");
};
template <>
class WindowHandler<WindowType::Window2D> : public BaseWindowHandler {
 public:
  explicit WindowHandler<WindowType::Window2D>(uint32_t id)
      : BaseWindowHandler(id) {
    scene_info_.set_width(800);
    scene_info_.set_height(600);
    roadstar::common::Point2D *pos = scene_info_.mutable_scene_rect_pos();
    pos->set_x(0);
    pos->set_y(0);
  }
  template <Object2DType::Type Type>
  std::unique_ptr<Object2DHandler<Type>> Build2D();
  template <Object2DType::Type Type, typename... Args>
  std::unique_ptr<Object2DHandler<Type>> Build2D(Args &&... args);
  template <Object2DType::Type Type, typename... Args>
  void Create2D(Args &&... args);
  void SetSceneInfo(const SceneInfo &info);
  void Render(const SceneInfo &info);
  void Render();

 private:
  SceneInfo scene_info_;
  WindowRenderRequest GetRequest(bool change_view);
};

template <>
class WindowHandler<WindowType::Window3D> : public BaseWindowHandler {
 public:
  explicit WindowHandler<WindowType::Window3D>(uint32_t id)
      : BaseWindowHandler(id) {
    camera_info_.set_field_of_view(45);
    Vector3D *pos = camera_info_.mutable_pos();
    pos->set_x(0);
    pos->set_y(0);
    pos->set_z(50);
    Vector3D *up_vector = camera_info_.mutable_up_vector();
    up_vector->set_x(1);
    up_vector->set_y(0);
    up_vector->set_z(0);
    Vector3D *view_center = camera_info_.mutable_view_center();
    view_center->set_x(0);
    view_center->set_y(0);
    view_center->set_z(0);
  }
  template <Object3DType::Type Type>
  std::unique_ptr<Object3DHandler<Type>> Build3D();
  template <Object3DType::Type Type, typename... Args>
  std::unique_ptr<Object3DHandler<Type>> Build3D(Args &&... args);
  template <Object3DType::Type Type, typename... Args>
  void Create3D(Args &&... args);
  void SetCameraInfo(const CameraInfo3D info);
  void Render(const CameraInfo3D &info);
  void Render();

 private:
  CameraInfo3D camera_info_;
  WindowRenderRequest GetRequest(bool change_view);
};

using Window2DHandler = WindowHandler<WindowType::Window2D>;
using Window2DHandlerPtr = std::shared_ptr<Window2DHandler>;
using Window3DHandler = WindowHandler<WindowType::Window3D>;
using Window3DHandlerPtr = std::shared_ptr<Window3DHandler>;
using PointCloudHandler = Object3DHandler<Object3DType::Pointcloud>;
using PointCloudHandlerPtr = std::unique_ptr<PointCloudHandler>;
using PlaneHandler = Object3DHandler<Object3DType::Plane>;
using PlaneHandlerPtr = std::unique_ptr<PlaneHandler>;
using SphereHandler = Object3DHandler<Object3DType::Sphere>;
using SphereHandlerPtr = std::unique_ptr<SphereHandler>;
using Text3DHandler = Object3DHandler<Object3DType::Text>;
using Text3DHandlerPtr = std::unique_ptr<Text3DHandler>;
using Points3DHandler = Object3DHandler<Object3DType::Points>;
using Points3DHandlerPtr = std::unique_ptr<Points3DHandler>;
using Points2DHandler = Object2DHandler<Object2DType::Points>;
using Points2DHandlerPtr = std::unique_ptr<Points2DHandler>;
using Line2DHandler = Object2DHandler<Object2DType::Line>;
using Line2DHandlerPtr = std::unique_ptr<Line2DHandler>;
using Polygon2DHandler = Object2DHandler<Object2DType::Polygon>;
using Polygon2DHandlerPtr = std::unique_ptr<Polygon2DHandler>;
using Image2DHandler = Object2DHandler<Object2DType::Image>;
using Image2DHandlerPtr = std::unique_ptr<Image2DHandler>;
using Chart2DHandler = Object2DHandler<Object2DType::Chart>;
using Chart2DHandlerPtr = std::unique_ptr<Chart2DHandler>;
using Text2DHandler = Object2DHandler<Object2DType::Text>;
using Text2DHandlerPtr = std::unique_ptr<Text2DHandler>;

class RemoteVisualizer {
 public:
  template <WindowType::Type Type>
  std::shared_ptr<WindowHandler<Type>> CreateWindow(
      std::string window_name = "");

 private:
  friend class WindowHandler<WindowType::Window3D>;
  friend class WindowHandler<WindowType::Window2D>;
  friend BaseWindowHandler;
  friend BaseObjectHandler;
  friend class BaseObject2DHandler;
  friend class BaseObject3DHandler;
  uint32_t Create3DObject(Object3DType::Type type, uint32_t wid);
  uint32_t Create2DObject(Object2DType::Type type, uint32_t wid);

  bool DeleteWindow(uint32_t wid);
  bool DeleteObject(uint32_t wid, uint32_t id);

  bool RenderObject(BaseObject3DHandler *);
  bool DrawObject2D(BaseObject2DHandler *);
  bool WindowRender(WindowRenderRequest request);

  uint32_t CreateWindow(WindowType::Type type, std::string name);
  ~RemoteVisualizer() = default;

 private:
  std::unique_ptr<RemoteVisualizerService::Stub> stub_ = nullptr;
  uint32_t connect_failed_times_ = 0;
  uint32_t GRPCConnect(
      bool is_judged_by_oid,
      std::function<grpc::Status(grpc::ClientContext *context, Response *reply)>
          callback);
  DECLARE_SINGLETON(RemoteVisualizer);
};
inline uint32_t BaseObjectHandler::GetParentId() const {
  return parent_->GetId();
}

template <Object3DType::Type Type>
std::unique_ptr<Object3DHandler<Type>>
WindowHandler<WindowType::Window3D>::Build3D() {
  if (auto id = RemoteVisualizer::instance()->Create3DObject(Type, id_); id) {
    return std::make_unique<Object3DHandler<Type>>(id, shared_from_this());
  }
  return nullptr;
}

template <Object3DType::Type Type, typename... Args>
std::unique_ptr<Object3DHandler<Type>>
WindowHandler<WindowType::Window3D>::Build3D(Args &&... args) {
  auto handler = Build3D<Type>();
  if (handler) {
    handler->SetData(std::forward<Args>(args)...);
    handler->Update(true);
  }
  return handler;
}

template <Object3DType::Type Type, typename... Args>
void WindowHandler<WindowType::Window3D>::Create3D(Args &&... args) {
  auto handler = Build3D<Type>();
  if (handler) {
    handler->SetData(std::forward<Args>(args)...);
    handler->Update(true);
    objects_.emplace_back(std::move(handler));
  }
}

template <Object2DType::Type Type>
std::unique_ptr<Object2DHandler<Type>>
WindowHandler<WindowType::Window2D>::Build2D() {
  if (auto id = RemoteVisualizer::instance()->Create2DObject(Type, id_); id) {
    return std::make_unique<Object2DHandler<Type>>(id, shared_from_this());
  }
  return nullptr;
}
template <Object2DType::Type Type, typename... Args>
std::unique_ptr<Object2DHandler<Type>>
WindowHandler<WindowType::Window2D>::Build2D(Args &&... args) {
  auto handler = Build2D<Type>();
  if (handler) {
    handler->SetData(std::forward<Args>(args)...);
    handler->Update(true);
  }
  return handler;
}

template <Object2DType::Type Type, typename... Args>
void WindowHandler<WindowType::Window2D>::Create2D(Args &&... args) {
  auto handler = Build2D<Type>();
  if (handler) {
    handler->SetData(std::forward<Args>(args)...);
    handler->Update(true);
    objects_.emplace_back(std::move(handler));
  }
}
template <typename... Args>
void BasePointsObject2DHandler::SetData(std::vector<Point2D> points, int size,
                                        Args &&... args) {
  SetSize(size);
  SetPoints(points);
  (SetProporty(args, false), ...);
}

template <typename... Args>
void Object2DHandler<Object2DType::Image>::SetData(cv::Mat mat,
                                                   Args &&... args) {
  SetImage(std::move(mat));
  (SetProporty(args, false), ...);
}

template <typename... Args>
void Object2DHandler<Object2DType::Chart>::SetData(Chart chart,
                                                   Args &&... args) {
  SetChart(std::move(chart));
  (SetProporty(args, false), ...);
}
template <typename... Args>
void Object2DHandler<Object2DType::Text>::SetData(std::string text,
                                                  Args &&... args) {
  SetText(std::move(text));
  (SetProporty(args, false), ...);
}

template <typename... Args>
void Object3DHandler<Object3DType::Pointcloud>::SetData(
    roadstar::drivers::lidar::PointCloud point_cloud, Args &&... args) {
  SetPointCloud(point_cloud);
  (SetProporty(args, false), ...);
  Update(true);
}

template <typename... Args>
void Object3DHandler<Object3DType::Plane>::SetData(float width, float height,
                                                   Args &&... args) {
  SetPlane(width, height, false);
  (SetProporty(args, false), ...);
}
template <typename... Args>
void Object3DHandler<Object3DType::Sphere>::SetData(int rings, int slices,
                                                    float radius,
                                                    Args &&... args) {
  SetSphere(rings, slices, radius, false);
  (SetProporty(args, false), ...);
}
template <typename... Args>
void Object3DHandler<Object3DType::Text>::SetData(std::string context,
                                                  float depth,
                                                  Args &&... args) {
  SetText(context, depth, false);
  (SetProporty(args, false), ...);
}

template <typename... Args>
void Object3DHandler<Object3DType::Points>::SetData(
    const Points &points, Args &&... args, Object3DType::Type render_type) {
  SetPoints(points);
  SetRenderType(render_type);
  (SetProporty(args, false), ...);
}

inline void BaseObject3DHandler::SetObjectInfo(Object3D *object) const {
  object->set_id(id_);
  object->set_wid(GetParentId());
  *object->mutable_color() = color_;
  *object->mutable_transform() = transform_;
}

inline void BaseObject2DHandler::SetObjectInfo(Object2D *object) const {
  object->set_id(id_);
  object->set_wid(GetParentId());
  *object->mutable_color() = color_;
  *object->mutable_transform() = transform_;
}
inline void Object2DHandler<Object2DType::Image>::SetObjectInfo(
    Object2D *object) const {
  BaseObject2DHandler::SetObjectInfo(object);
  Transform2D transform = *object->mutable_transform();
  transform.set_scale(transform.scale() * basic_scale_);
  *object->mutable_transform() = std::move(transform);
}

inline void BasePointsObject2DHandler::SetObjectInfo(Object2D *object) const {
  BaseObject2DHandler::SetObjectInfo(object);
  Points2D *points = object->mutable_points();
  *points->mutable_data() = {points_.begin(), points_.end()};
  points->set_width(size_);
}

inline void BasePointsObject2DHandler::AddPoints(
    const std::vector<Point2D> &points) {
  points_.reserve(points_.size() + points.size());
  points_.insert(points_.end(), points.begin(), points.end());
}

template <WindowType::Type Type>
std::shared_ptr<WindowHandler<Type>> RemoteVisualizer::CreateWindow(
    std::string window_name) {
  if (auto id = CreateWindow(Type, window_name); id) {
    return std::make_shared<WindowHandler<Type>>(id);
  }
  return nullptr;
}

}  // namespace roadstar::common::visualization
#endif  // MODULES_COMMON_VISUALIZATION_REMOTE_VISUALIZER_REMOuTE_VISUALIZER_H_
