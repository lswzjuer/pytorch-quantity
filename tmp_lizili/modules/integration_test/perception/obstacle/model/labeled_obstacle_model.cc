
#include "modules/integration_test/perception/obstacle/model/labeled_obstacle_model.h"
#include <math.h>
#include "Eigen/Core"
#include "modules/common/log.h"
#include "modules/integration_test/perception/util/tool.h"
#include "modules/msgs/localization/proto/localization.pb.h"
#include "modules/msgs/perception/proto/obstacle.pb.h"

#define _USE_MATH_DEFINES

namespace roadstar {
namespace integration_test {

using Obstacle = roadstar::perception::Obstacle;
using Localization = roadstar::localization::Localization;

json LabeledObstacleModel::ToJson() const {
  json out = ObstacleBaseModel::ToJson();
  out["r"] = r_;
  out["id"] = id_;
  out["velocity"] = velocity_;
  out["heading"] = heading_;
  out["type"] = type_;
  return out;
}
void LabeledObstacleModel::FromJson(const json& value) {
  ObstacleBaseModel::FromJson(value);
  r_ = value["r"];
  id_ = value["id"];
  velocity_ = value["velocity"];
  heading_ = value["heading"];
  type_ = value["type"];
}

LabeledObstacleModel LabeledObstacleModel::ToEgoFront(
    const std::string& perception_version) {
  Eigen::Vector3d origin(x_, y_, z_);
  Eigen::Vector3d ret = Tool::Velodyne64ToEgoFront(origin, perception_version);
  LabeledObstacleModel model(ret[0] / 10, ret[1] / 10, ret[2] / 10, h_ / 10,
                             l_ / 10, w_ / 10, r_, id_, velocity_, heading_,
                             type_);
  return model;
}

LabeledObstacleModel LabeledObstacleModel::Velodyne64ToUtm(
    const std::string& perception_version, const LocationModel& location) {
  Eigen::Vector3d origin(x_ / 10, y_ / 10, z_ / 10);
  Localization localization;
  location.ToLocalizationMsg(&localization);
  Eigen::Vector3d ret =
      Tool::Velodyne64ToWorld(origin, localization, perception_version);
  LabeledObstacleModel model(ret[0], ret[1], ret[2], h_ / 10, l_ / 10, w_ / 10,
                             r_, id_, velocity_, heading_, type_);
  return model;
}

std::vector<PointENU> LabeledObstacleModel::GetVelodyneTypePolygon() const {
  //  To find the intersection,The vertices of the rectangle must be clockwise.
  std::vector<PointENU> polygon;
  double theta = r_ - M_PI / 2;
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

std::vector<PointENU> LabeledObstacleModel::GetEgoFrontTypePolygon(
    const std::string& version) const {
  std::vector<PointENU> polygon = GetVelodyneTypePolygon();
  for (auto& it : polygon) {
    Eigen::Vector3d origin(it.x(), it.y(), 0);
    Eigen::Vector3d ret = Tool::Velodyne64ToEgoFront(origin, version);
    it.set_x(ret[0] / 10);
    it.set_y(ret[1] / 10);
  }
  return polygon;
}

LabeledObstacleModel* LabeledObstacleModel::SetX(double x) {
  x_ = x;
  return this;
}
LabeledObstacleModel* LabeledObstacleModel::SetY(double y) {
  y_ = y;
  return this;
}

LabeledObstacleModel* LabeledObstacleModel::SetZ(double z) {
  z_ = z;
  return this;
}

LabeledObstacleModel* LabeledObstacleModel::SetH(double h) {
  h_ = h;
  return this;
}

LabeledObstacleModel* LabeledObstacleModel::SetL(double l) {
  l_ = l;
  return this;
}

LabeledObstacleModel* LabeledObstacleModel::SetW(double w) {
  w_ = w;
  return this;
}

LabeledObstacleModel* LabeledObstacleModel::SetR(double r) {
  r_ = r;
  return this;
}

double LabeledObstacleModel::GetR() const {
  return r_;
}

LabeledObstacleModel* LabeledObstacleModel::SetId(size_t id) {
  id_ = id;
  return this;
}

LabeledObstacleModel* LabeledObstacleModel::SetVelocity(double velocity) {
  velocity_ = velocity;
  return this;
}

LabeledObstacleModel* LabeledObstacleModel::SetHeading(double heading) {
  heading_ = heading;
  return this;
}

LabeledObstacleModel* LabeledObstacleModel::SetType(int type) {
  type_ = type;
  return this;
}

size_t LabeledObstacleModel::GetId() const {
  return id_;
}

double LabeledObstacleModel::GetVelocity() const {
  return velocity_;
}

double LabeledObstacleModel::GetHeading() const {
  return heading_;
}

int LabeledObstacleModel::GetType() const {
  return type_;
}

}  // namespace integration_test
}  // namespace roadstar
