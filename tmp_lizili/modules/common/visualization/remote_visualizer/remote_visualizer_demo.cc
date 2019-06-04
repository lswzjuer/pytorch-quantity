#include <csignal>
#include <random>
#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "modules/common/adapters/adapter.h"
#include "modules/common/adapters/adapter_manager.h"
#include "modules/common/log.h"
#include "modules/common/util/file.h"
#include "modules/common/visualization/remote_visualizer/remote_visualizer.h"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"

namespace roadstar::common::visualization {

using common::visualization::RemoteVisualizer;
using roadstar::common::adapter::Adapter;
using roadstar::common::adapter::AdapterConfig;
using roadstar::common::adapter::AdapterManager;
using roadstar::common::adapter::AdapterManagerConfig;
using roadstar::common::util::GetProtoFromASCIIFile;
using roadstar::common::util::GetProtoFromFile;
using roadstar::common::util::SetProtoToASCIIFile;

class RemoteVisualizerDemo {
 public:
  RemoteVisualizerDemo() {
    // Create Windows
    window_handler =
        visualizer.CreateWindow<WindowType::Window3D>("Pointcloud");
    window_handler2 =
        visualizer.CreateWindow<WindowType::Window3D>("Window3D-Test");
    window_handler3 =
        visualizer.CreateWindow<WindowType::Window2D>("Window2D-Test");
    window_handler4 =
        visualizer.CreateWindow<WindowType::Window2D>("Chart-Test");
    // Create Object
    if (window_handler != nullptr && window_handler2 != nullptr &&
        window_handler3 != nullptr&& window_handler4 != nullptr) {
      pointcloud_handler = window_handler->Build3D<Object3DType::Pointcloud>();
      Vector3D translation;
      translation.set_x(0);
      translation.set_y(0);
      translation.set_z(-2);
      Rotation rotation;
      auto vector = rotation.mutable_vector();
      vector->set_x(1);
      vector->set_y(0);
      vector->set_z(0);
      rotation.set_angle(90);
      Color color;
      color.set_red(255);
      color.set_blue(255);
      color.set_green(255);
      plane_handler = window_handler2->Build3D<Object3DType::Plane>(
          30, 30, color, rotation, translation);
      point2_handler = window_handler3->Build2D<Object2DType::Points>();
      color.set_blue(0);
      color.set_green(0);
      vector->set_x(0);
      vector->set_z(1);
      text_handler = window_handler2->Build3D<Object3DType::Text>(
          "Only for test!", 1, color, rotation);
      sphere_handler =
          window_handler2->Build3D<Object3DType::Sphere>(20, 20, 3, color);
      window_handler2->Render();
      points_handler = window_handler2->Build3D<Object3DType::Points>();
      sphere_handler = window_handler2->Build3D<Object3DType::Sphere>();
      point2_handler = window_handler3->Build2D<Object2DType::Points>();
      line2_handler = window_handler3->Build2D<Object2DType::Line>();
      image_handler = window_handler3->Build2D<Object2DType::Image>();
      chart_handler = window_handler4->Build2D<Object2DType::Chart>();
      text2_handler = window_handler4->Build2D<Object2DType::Text>("Testing");
    }
    if (pointcloud_handler == nullptr || plane_handler == nullptr ||
        points_handler == nullptr || sphere_handler == nullptr ||
        text_handler == nullptr || point2_handler == nullptr ||
        line2_handler == nullptr || image_handler == nullptr|| chart_handler == nullptr|| text2_handler == nullptr) {
      AINFO << "Create object Failed!";
    }
  }

  void Test();

  RemoteVisualizer &visualizer{*RemoteVisualizer::instance()};
  Window3DHandlerPtr window_handler;
  Window3DHandlerPtr window_handler2;
  Window2DHandlerPtr window_handler3;
  Window2DHandlerPtr window_handler4;

  PointCloudHandlerPtr pointcloud_handler;
  PlaneHandlerPtr plane_handler;
  Points3DHandlerPtr points_handler;
  SphereHandlerPtr sphere_handler;
  Text3DHandlerPtr text_handler;

  Points2DHandlerPtr point2_handler;
  Line2DHandlerPtr line2_handler;
  Image2DHandlerPtr image_handler;
  Chart2DHandlerPtr chart_handler;
  Text2DHandlerPtr text2_handler;
};

void RemoteVisualizerDemo::Test() {
  CameraInfo3D info;
  info.set_field_of_view(45);
  Vector3D *position = info.mutable_pos();
  position->set_x(0);
  position->set_y(0);
  position->set_z(80);
  Vector3D *up_vector = info.mutable_up_vector();
  up_vector->set_x(1);
  up_vector->set_y(0);
  up_vector->set_z(0);
  Vector3D *view_center = info.mutable_view_center();
  view_center->set_x(0);
  view_center->set_y(0);
  view_center->set_z(0);

  
  Color color2;
  color2.set_red(255);
  color2.set_green(255);
  color2.set_blue(0);
  for (auto i = 1; i < 100; i++) {
    window_handler2->Create3D<Object3DType::Sphere>(20, 20, float(i) / 5,
                                                    color2);
    window_handler2->Render();
  }

  
  // 2D Window
  // Draw Chart
  Chart chart;
  chart.set_title("Line chart");
  chart.set_width(800);
  chart.set_height(600);
  chart.set_type(ChartType::Line);
  std::random_device rd;
  std::uniform_real_distribution<double> rand_num(0, 20);
  for (int i = 0; i < 3; i++) {
    DataList *data_list = chart.add_data();
    data_list->set_tag(std::to_string(i));
    for (int j = 0; j < 50; j++) {
      auto p = data_list->add_points();
      p->set_x(j);
      p->set_y(rand_num(rd));
    }
  }
  chart_handler->SetChart(chart);
  window_handler4->Render();

  // Draw Points
  std::vector<Point2D> points_2d;

  std::uniform_real_distribution<double> pos(0, 500);

  for (int i = 0; i < 10000; i++) {
    Point2D p;
    p.set_x(pos(rd));
    p.set_y(pos(rd));
    points_2d.emplace_back(std::move(p));
  }
  point2_handler->AddPoints(std::move(points_2d));
  Color color;
  color.set_blue(0);
  color.set_green(0);
  color.set_red(0);
  point2_handler->SetProporty(color);
  window_handler3->Render();

  color.set_red(255);
  // Draw LineLoop
  line2_handler->SetProporty(color, false);
  line2_handler->SetSize(1, false);
  line2_handler->AddPoint(250, 250, false);
  line2_handler->AddPoint(500, 250, false);
  line2_handler->AddPoint(500, 500, false);
  line2_handler->AddPoint(250, 500, true);
 

  SceneInfo sinfo;
  sinfo.set_width(800);
  sinfo.set_height(800);
  sinfo.set_rotation(45);
  roadstar::common::Point2D *position2 = sinfo.mutable_scene_rect_pos();
  position2->set_x(0);
  position2->set_y(0);
  window_handler3->Render(sinfo);

  // Draw Pointcloud
  AdapterManager::AddPointCloudCallback([this](const auto &msg) {
    roadstar::drivers::lidar::PointCloud pointcloud;
    AINFO << "Start!";
    for (auto const &pt : msg.points) {
      auto *p = pointcloud.add_points();
      p->set_x(pt.x);
      p->set_y(pt.y);
      p->set_z(pt.z);
      p->set_intensity(pt.intensity);
    }
    if (pointcloud_handler != nullptr) {
      // pointcloud_handler->SetRotation(0, 1, 0, 100);
      pointcloud_handler->SetPointCloud(std::move(pointcloud));
      window_handler->Render();
    }
  });
}
}  // namespace roadstar::common::visualization

void SigintHandler(int signal_num) {
  AINFO << "Received signal: " << signal_num;
  if (signal_num != SIGINT) {
    return;
  }
  bool static is_stopping = false;
  if (is_stopping) {
    return;
  }
  is_stopping = true;
  ros::shutdown();
}

int main(int argc, char *argv[]) {
  // google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  signal(SIGINT, SigintHandler);
  ros::init(argc, argv, "remote_visualizer_test");
  ros::AsyncSpinner spinner(2);
  roadstar::common::adapter::AdapterManagerConfig adapter_manager_config;

  auto config = adapter_manager_config.add_config();
  config->set_type(roadstar::common::adapter::AdapterConfig::POINT_CLOUD);
  config->set_mode(roadstar::common::adapter::AdapterConfig::RECEIVE_ONLY);
  adapter_manager_config.set_is_ros(true);
  roadstar::common::adapter::AdapterManager::InitAdapters(
      adapter_manager_config);
  roadstar::common::visualization::RemoteVisualizerDemo demo;
  demo.Test();
  spinner.start();
  ros::waitForShutdown();
  return 0;
}
