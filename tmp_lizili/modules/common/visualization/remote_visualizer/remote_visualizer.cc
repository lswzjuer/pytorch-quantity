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
#include "modules/common/visualization/remote_visualizer/remote_visualizer.h"
#include <cstdint>
#include <opencv2/opencv.hpp>
#include <utility>
#include "gflags/gflags.h"
#include "modules/common/log.h"
#include "modules/common/visualization/remote_visualizer/proto/remote_visualizer.pb.h"
#include "modules/common/visualization/remote_visualizer/proto/remote_visualizer_grpc.pb.h"
#include "modules/common/visualization/remote_visualizer/remote_visualizer_gflags.h"
using google::protobuf::RepeatedPtrField;
using grpc::Channel;
using grpc::ClientContext;

namespace roadstar::common::visualization {

void FillClientContext(ClientContext *context) {
  if (FLAGS_remote_visualizer_timeout > 0) {
    std::chrono::system_clock::time_point deadline =
        std::chrono::system_clock::now() +
        std::chrono::milliseconds(FLAGS_remote_visualizer_timeout);
    context->set_deadline(deadline);
  }
}

uint32_t RemoteVisualizer::GRPCConnect(
    bool is_judged_by_oid,
    std::function<grpc::Status(grpc::ClientContext *context, Response *reply)>
        callback) {
  if (connect_failed_times_ > 2) {
    return 0;
  }
  Response reply;
  grpc::ClientContext context;
  FillClientContext(&context);
  grpc::Status status = callback(&context, &reply);
  int judge_id = is_judged_by_oid ? reply.oid() : reply.wid();
  if (status.ok() && judge_id != 0) {
    return judge_id;
  } else {
    if (!status.ok()) {  // GRPC connection failed
      connect_failed_times_++;
      AERROR << "GRPC connection failed. error_code: " << status.error_code()
             << " error_msg: " << status.error_message();
      if (connect_failed_times_ > 2) {
        AERROR << "Multiple GRPC connection failure happened. Server may have "
                  "died.";
      }
    } else {
      AERROR << reply.msg();
    }
  }
  return 0;
}

RemoteVisualizer::RemoteVisualizer()
    : stub_(RemoteVisualizerService::NewStub(
          grpc::CreateChannel(FLAGS_remote_visualizer_address,
                              grpc::InsecureChannelCredentials()))) {}

uint32_t RemoteVisualizer::Create3DObject(Object3DType::Type type,
                                          uint32_t wid) {
  CreateObjectRequest create_request;
  create_request.set_type3d(type);
  create_request.set_wid(wid);
  return GRPCConnect(true, [&](grpc::ClientContext *context, Response *reply) {
    return stub_->CreateObject(context, create_request, reply);
  });
}

uint32_t RemoteVisualizer::Create2DObject(Object2DType::Type type,
                                          uint32_t wid) {
  CreateObjectRequest create_request;
  create_request.set_type2d(type);
  create_request.set_wid(wid);
  return GRPCConnect(true, [&](grpc::ClientContext *context, Response *reply) {
    return stub_->CreateObject(context, create_request, reply);
  });
}

bool RemoteVisualizer::RenderObject(BaseObject3DHandler *handler) {
  return GRPCConnect(true, [&](grpc::ClientContext *context, Response *reply) {
    return stub_->Render(context, handler->GetObject3D(), reply);
  });
}

bool RemoteVisualizer::DrawObject2D(BaseObject2DHandler *handler) {
  return GRPCConnect(true, [&](grpc::ClientContext *context, Response *reply) {
    return stub_->Draw(context, handler->GetObject2D(), reply);
  });
}

bool RemoteVisualizer::WindowRender(WindowRenderRequest request) {
  return GRPCConnect(false, [&](grpc::ClientContext *context, Response *reply) {
    return stub_->WindowRender(context, std::move(request), reply);
  });
}

bool BaseObject2DHandler::Update(bool is_update) {
  if (is_update) {
    return RemoteVisualizer::instance()->DrawObject2D(this);
  }
  return true;
}
bool BaseObject3DHandler::Update(bool is_update) {
  if (is_update) {
    return RemoteVisualizer::instance()->RenderObject(this);
  }
  return true;
}
void WindowHandler<WindowType::Window3D>::SetCameraInfo(
    const CameraInfo3D info) {
  camera_info_ = info;
}
void WindowHandler<WindowType::Window3D>::Render(const CameraInfo3D &info) {
  SetCameraInfo(info);
  RemoteVisualizer::instance()->WindowRender(GetRequest(true));
  objects_.clear();
}
void WindowHandler<WindowType::Window3D>::Render() {
  RemoteVisualizer::instance()->WindowRender(GetRequest(false));
  objects_.clear();
}
void WindowHandler<WindowType::Window2D>::SetSceneInfo(const SceneInfo &info) {
  scene_info_ = info;
}
void WindowHandler<WindowType::Window2D>::Render(const SceneInfo &info) {
  SetSceneInfo(info);
  RemoteVisualizer::instance()->WindowRender(GetRequest(true));
  objects_.clear();
}
void WindowHandler<WindowType::Window2D>::Render() {
  RemoteVisualizer::instance()->WindowRender(GetRequest(false));
  objects_.clear();
}
WindowRenderRequest WindowHandler<WindowType::Window3D>::GetRequest(
    bool change_view) {
  WindowRenderRequest request;
  request.set_wid(id_);
  request.set_change_view(change_view);
  if (change_view) {
    *request.mutable_camera_info3d() = camera_info_;
  }
  return request;
}
WindowRenderRequest WindowHandler<WindowType::Window2D>::GetRequest(
    bool change_view) {
  WindowRenderRequest request;
  request.set_wid(id_);
  request.set_change_view(change_view);
  if (change_view) {
    *request.mutable_scene_info() = scene_info_;
  }
  return request;
}
void BasePointsObject2DHandler::SetPoints(std::vector<Point2D> points,
                                          bool is_update) {
  points_ = std::move(points);
  Update(is_update);
}
Object2D Object2DHandler<Object2DType::Points>::GetObject2D() const {
  Object2D object;
  object.set_type(Object2DType::Points);
  SetObjectInfo(&object);
  return object;
}

Object2D Object2DHandler<Object2DType::Line>::GetObject2D() const {
  Object2D object;
  object.set_type(Object2DType::Line);
  SetObjectInfo(&object);
  return object;
}

Object2D Object2DHandler<Object2DType::Polygon>::GetObject2D() const {
  Object2D object;
  object.set_type(Object2DType::Polygon);
  SetObjectInfo(&object);
  return object;
}

Object2D Object2DHandler<Object2DType::Image>::GetObject2D() const {
  Object2D object;
  object.set_type(Object2DType::Image);
  *object.mutable_image() = image_;
  SetObjectInfo(&object);
  return object;
}
Object2D Object2DHandler<Object2DType::Chart>::GetObject2D() const {
  Object2D object;
  object.set_type(Object2DType::Chart);
  *object.mutable_chart() = chart_;
  SetObjectInfo(&object);
  return object;
}
Object2D Object2DHandler<Object2DType::Text>::GetObject2D() const {
  Object2D object;
  object.set_type(Object2DType::Text);
  object.set_text(text_);
  SetObjectInfo(&object);
  return object;
}
Object3D Object3DHandler<Object3DType::Pointcloud>::GetObject3D() const {
  Object3D object;
  object.set_type3d(Object3DType::Pointcloud);
  *object.mutable_pointcloud() = point_cloud_;
  SetObjectInfo(&object);
  return object;
}

Object3D Object3DHandler<Object3DType::Plane>::GetObject3D() const {
  Object3D object;
  object.set_type3d(Object3DType::Plane);
  *object.mutable_plane() = plane_;
  SetObjectInfo(&object);
  return object;
}

Object3D Object3DHandler<Object3DType::Points>::GetObject3D() const {
  Object3D object;
  object.set_type3d(type_);
  *object.mutable_points() = points_;
  SetObjectInfo(&object);
  return object;
}

Object3D Object3DHandler<Object3DType::Sphere>::GetObject3D() const {
  Object3D object;
  object.set_type3d(Object3DType::Sphere);
  *object.mutable_sphere() = sphere_;
  SetObjectInfo(&object);
  return object;
}

Object3D Object3DHandler<Object3DType::Text>::GetObject3D() const {
  Object3D object;
  object.set_type3d(Object3DType::Text);
  *object.mutable_text() = text_;
  SetObjectInfo(&object);
  return object;
}
bool Object2DHandler<Object2DType::Image>::SetImage(cv::Mat mat,
                                                    bool is_update) {
  if (mat.type() != CV_8UC4 && mat.type() != CV_8UC3 && mat.type() != CV_8UC2 &&
      mat.type() != CV_8UC1) {
    mat.convertTo(mat, CV_8U);
  }
  basic_scale_ = mat.cols * mat.rows * mat.channels() / 500000.0;
  if (basic_scale_ > 1) {
    cv::Mat small;
    cv::resize(mat, small, {0, 0}, 1 / basic_scale_, 1 / basic_scale_);
    mat = small;
  } else {
    basic_scale_ = 1.0;
  }
  std::string str;
  str.assign(reinterpret_cast<const char *>(mat.data),
             mat.total() * mat.elemSize());
  image_.set_data(std::move(str));
  image_.set_width(mat.cols);
  image_.set_height(mat.rows);
  image_.set_step(mat.step);
  switch (mat.type()) {
    case CV_8UC1:
      image_.set_pixel_format(drivers::TYPE_8UC1);
      break;
    case CV_8UC2:
      image_.set_pixel_format(drivers::TYPE_8UC2);
      break;
    case CV_8UC3:
      image_.set_pixel_format(drivers::TYPE_8UC3);
      break;
    case CV_8UC4:
      image_.set_pixel_format(drivers::TYPE_8UC4);
      break;
  }
  Update(is_update);
  return true;
}
void Object2DHandler<Object2DType::Chart>::SetChart(const Chart &chart,
                                                    bool is_update) {
  chart_ = chart;
  Update(is_update);
}
void Object2DHandler<Object2DType::Chart>::SetChartType(ChartType::Type type,
                                                        bool is_update) {
  chart_.set_type(std::move(type));
  Update(is_update);
}
void Object2DHandler<Object2DType::Text>::SetText(const std::string &text,
                                                  bool is_update) {
  text_ = text;
  Update(is_update);
}
void Object3DHandler<Object3DType::Pointcloud>::SetPointCloud(
    roadstar::drivers::lidar::PointCloud point_cloud, bool is_update) {
  point_cloud_ = std::move(point_cloud);
  Update(is_update);
}
void Object3DHandler<Object3DType::Plane>::SetPlane(float width, float height,
                                                    bool is_update) {
  plane_.set_width(width);
  plane_.set_height(height);
  Update(is_update);
}

void Object3DHandler<Object3DType::Points>::SetPoints(const Points &points,
                                                      bool is_update) {
  points_ = points;
  Update(is_update);
}

bool Object3DHandler<Object3DType::Points>::SetRenderType(
    Object3DType::Type type, bool is_update) {
  if (type != Object3DType::Points && type != Object3DType::Lines &&
      type != Object3DType::LineLoop && type != Object3DType::LineStrip &&
      type != Object3DType::Triangles) {
    return false;
  }
  type_ = type;
  Update(is_update);
  return true;
}

void Object3DHandler<Object3DType::Sphere>::SetSphere(int rings, int slices,
                                                      float radius,
                                                      bool is_update) {
  sphere_.set_rings(rings);
  sphere_.set_slices(slices);
  sphere_.set_radius(radius);
  Update(is_update);
}
void Object3DHandler<Object3DType::Text>::SetText(std::string context,
                                                  float depth, bool is_update) {
  text_.set_depth(depth);
  text_.set_context(context);
  Update(is_update);
}
uint32_t RemoteVisualizer::CreateWindow(WindowType::Type type,
                                        std::string name) {
  CreateWinRequest create_request;
  create_request.set_type(type);
  create_request.set_window_name(std::move(name));
  return GRPCConnect(false, [&](grpc::ClientContext *context, Response *reply) {
    grpc::Status status = stub_->CreateWin(context, create_request, reply);
    return status;
  });
}

bool RemoteVisualizer::DeleteWindow(uint32_t wid) {
  DeleteRequest delete_request;
  delete_request.set_wid(wid);
  return GRPCConnect(false, [&](grpc::ClientContext *context, Response *reply) {
    return stub_->DeleteWindow(context, delete_request, reply);
  });
}
bool RemoteVisualizer::DeleteObject(uint32_t wid, uint32_t id) {
  DeleteRequest delete_request;
  delete_request.set_wid(wid);
  delete_request.set_oid(id);
  return GRPCConnect(true, [&](grpc::ClientContext *context, Response *reply) {
    return stub_->DeleteObject(context, std::move(delete_request), reply);
  });
}
void BaseObjectHandler::SetProporty(const Color &color, bool is_update) {
  color_ = color;
  Update(is_update);
}

void BaseObject3DHandler::SetProporty(float scale, bool is_update) {
  transform_.set_scale(scale);
  Update(is_update);
}
void BaseObject3DHandler::SetProporty(const Translation &translation,
                                      bool is_update) {
  *transform_.mutable_translation() = translation;
  Update(is_update);
}
void BaseObject3DHandler::SetProporty(const Rotation &rotation,
                                      bool is_update) {
  *transform_.mutable_rotation() = rotation;
  Update(is_update);
}
void BaseObject2DHandler::SetProporty(double scale, bool is_update) {
  transform_.set_scale(scale);
  Update(is_update);
}
void BaseObject2DHandler::SetProporty(const Translation &translation,
                                      bool is_update) {
  *transform_.mutable_pos() = translation;
  Update(is_update);
}
void BaseObject2DHandler::SetProporty(const Rotation2D &rotation,
                                      bool is_update) {
  *transform_.mutable_rotation() = rotation;
  Update(is_update);
}

void BasePointsObject2DHandler::AddPoint(double x, double y, bool is_update) {
  Point2D p;
  p.set_x(x);
  p.set_y(y);
  points_.emplace_back(std::move(p));
  Update(is_update);
}

void BasePointsObject2DHandler::ClearPoints() {
  points_.clear();
}

BaseWindowHandler::~BaseWindowHandler() {
  RemoteVisualizer::instance()->DeleteWindow(id_);
}

BaseObjectHandler::~BaseObjectHandler() {
  RemoteVisualizer::instance()->DeleteObject(GetParentId(), id_);
}

}  // namespace roadstar::common::visualization
