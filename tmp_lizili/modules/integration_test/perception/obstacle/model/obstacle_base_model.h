#ifndef MODULES_INTEGRATION_TEST_PERCEPTION_OBSTACLE_MODEL_OBSTACLE_BASE_MODEL_H
#define MODULES_INTEGRATION_TEST_PERCEPTION_OBSTACLE_MODEL_OBSTACLE_BASE_MODEL_H

#include "third_party/json/json.hpp"

namespace roadstar {
namespace integration_test {

using json = nlohmann::json;

class ObstacleBaseModel {
 public:
  ObstacleBaseModel();
  ObstacleBaseModel(double x, double y, double z, double l, double h, double w);
  double GetX() const;
  double GetY() const;
  double GetZ() const;
  double GetH() const;
  double GetL() const;
  double GetW() const;

  virtual json ToJson() const;
  virtual void FromJson(const json& out);

 protected:
  double x_;
  double y_;
  double z_;
  double l_;
  double h_;
  double w_;
};

}  // namespace integration_test
}  // namespace roadstar

#endif
