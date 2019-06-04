#include "modules/integration_test/perception/obstacle/model/obstacle_base_model.h"

namespace roadstar {
namespace integration_test {

ObstacleBaseModel::ObstacleBaseModel() {
  x_ = 0;
  y_ = 0;
  z_ = 0;
  l_ = 0;
  h_ = 0;
  w_ = 0;
}

ObstacleBaseModel::ObstacleBaseModel(double x, double y, double z, double l,
                                     double h, double w)
    : x_(x), y_(y), z_(z), l_(l), h_(h), w_(w) {}

double ObstacleBaseModel::GetX() const {
  return x_;
}

double ObstacleBaseModel::GetY() const {
  return y_;
}

double ObstacleBaseModel::GetZ() const {
  return z_;
}

double ObstacleBaseModel::GetH() const {
  return h_;
}

double ObstacleBaseModel::GetL() const {
  return l_;
}

double ObstacleBaseModel::GetW() const {
  return w_;
}

json ObstacleBaseModel::ToJson() const {
  json out;
  out["x"] = x_;
  out["y"] = y_;
  out["z"] = z_;
  out["l"] = l_;
  out["h"] = h_;
  out["w"] = w_;
  return out;
}

void ObstacleBaseModel::FromJson(const json& out) {
  x_ = out["x"];
  y_ = out["y"];
  z_ = out["z"];
  l_ = out["l"];
  h_ = out["h"];
  w_ = out["w"];
}

}  // namespace integration_test
}  // namespace roadstar
