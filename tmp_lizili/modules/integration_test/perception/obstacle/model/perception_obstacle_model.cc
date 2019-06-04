
#include "modules/integration_test/perception/obstacle/model/perception_obstacle_model.h"
#include "Eigen/Core"
#include "modules/common/log.h"
#include "modules/integration_test/perception/util/tool.h"
#include "modules/msgs/localization/proto/localization.pb.h"

namespace roadstar {
namespace integration_test {

using Localization = roadstar::localization::Localization;

json PerceptionObstacleModel::ToJson() const {
  json out = ObstacleBaseModel::ToJson();
  out["sensor_source"] = sensor_source_;
  out["heading"] = heading_;
  out["theta"] = theta_;
  out["velocity"] = velocity_;
  out["type"] = type_;
  return out;
}

void PerceptionObstacleModel::FromJson(const json& value) {
  ObstacleBaseModel::FromJson(value);
  sensor_source_ = value["sensor_source"];
  heading_ = value["heading"];
  theta_ = value["theta"];
  velocity_ = value["velocity"];
  type_ = value["type"];
}

void PerceptionObstacleModel::FromObstacleMsg(const Obstacle& ob) {
  x_ = ob.position().x();
  y_ = ob.position().y();
  z_ = 0;
  theta_ = ob.theta();
  l_ = ob.length();
  w_ = ob.width();
  h_ = ob.height();
  heading_ = ob.heading();
  sensor_source_ = ob.sensor_source();
  velocity_ = ob.velocity();
  type_ = ob.object_type();
}

PerceptionObstacleModel PerceptionObstacleModel::TransformToEgoFront(
    const LocationModel& model, const std::string& version) {
  PerceptionObstacleModel ob_model = *this;
  Eigen::Vector3d origin(x_, y_, z_);
  Localization locate;
  model.ToLocalizationMsg(&locate);
  Eigen::Vector3d ret = Tool::WorldToEgoFront(origin, locate, version);
  (&ob_model)->SetX(ret[0])->SetY(ret[1])->SetZ(ret[2]);
  return ob_model;
}

std::vector<PointENU> PerceptionObstacleModel::GetUtmTypePolygon() const {
  std::vector<PointENU> polygon;
  double theta = theta_;
  polygon.clear();
  std::vector<Eigen::Vector2d> points;
  double l_half = l_ / 2;
  double w_half = w_ / 2;
  Eigen::Vector2d center(x_, y_);
  Eigen::Vector2d v(l_half * cos(theta), l_half * sin(theta));
  Eigen::Vector2d u(-w_half * sin(theta), w_half * cos(theta));
  // cornet points
  Eigen::Vector2d pt_left_top = center + v + u;
  Eigen::Vector2d pt_left_bottom = center - v + u;
  Eigen::Vector2d pt_right_bottom = center - v - u;
  Eigen::Vector2d pt_right_top = center + v - u;
  points.push_back(pt_left_top);
  points.push_back(pt_right_top);
  points.push_back(pt_right_bottom);
  points.push_back(pt_left_bottom);
  points.push_back(pt_left_top);

  for (auto& it : points) {
    PointENU pt;
    pt.set_x(it[0]);
    pt.set_y(it[1]);
    polygon.push_back(pt);
  }

  return polygon;
}

std::vector<PointENU> PerceptionObstacleModel::GetEgoFrontTypePolygon(
    const LocationModel& model, const std::string& version) const {
  std::vector<PointENU> polygon = GetUtmTypePolygon();
  for (auto& it : polygon) {
    Eigen::Vector3d origin(it.x(), it.y(), it.z());
    Localization locate;
    model.ToLocalizationMsg(&locate);
    Eigen::Vector3d ret = Tool::WorldToEgoFront(origin, locate, version);
    it.set_x(ret[0]);
    it.set_y(ret[1]);
    it.set_z(ret[2]);
  }
  return polygon;
}

PerceptionObstacleModel* PerceptionObstacleModel::SetX(double x) {
  x_ = x;
  return this;
}
PerceptionObstacleModel* PerceptionObstacleModel::SetY(double y) {
  y_ = y;
  return this;
}

PerceptionObstacleModel* PerceptionObstacleModel::SetZ(double z) {
  z_ = z;
  return this;
}

PerceptionObstacleModel* PerceptionObstacleModel::SetH(double h) {
  h_ = h;
  return this;
}

PerceptionObstacleModel* PerceptionObstacleModel::SetL(double l) {
  l_ = l;
  return this;
}

PerceptionObstacleModel* PerceptionObstacleModel::SetW(double w) {
  w_ = w;
  return this;
}

PerceptionObstacleModel* PerceptionObstacleModel::SetSensorSource(
    int sensor_source) {
  sensor_source_ = sensor_source;
  return this;
}

PerceptionObstacleModel* PerceptionObstacleModel::SetTheta(double theta) {
  theta_ = theta;
  return this;
}

PerceptionObstacleModel* PerceptionObstacleModel::SetHeading(double heading) {
  heading_ = heading;
  return this;
}

PerceptionObstacleModel* PerceptionObstacleModel::SetVelocity(double velocity) {
  velocity_ = velocity;
  return this;
}

PerceptionObstacleModel* PerceptionObstacleModel::SetType(int type) {
  type_ = type;
  return this;
}

int PerceptionObstacleModel::GetSensorSource() const {
  return sensor_source_;
}
double PerceptionObstacleModel::GetTheta() const {
  return theta_;
}
double PerceptionObstacleModel::GetHeading() const {
  return heading_;
}

double PerceptionObstacleModel::GetVelocity() const {
  return velocity_;
}

int PerceptionObstacleModel::GetType() const {
  return type_;
}

}  // namespace integration_test
}  // namespace roadstar
