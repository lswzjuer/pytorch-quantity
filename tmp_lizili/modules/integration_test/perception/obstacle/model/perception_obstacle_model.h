#ifndef MODULES_INTEGRATION_TEST_PERCEPTION_OBSTACLE_MODEL_PERCEPTION_OBSTACLE_MODEL_H
#define MODULES_INTEGRATION_TEST_PERCEPTION_OBSTACLE_MODEL_PERCEPTION_OBSTACLE_MODEL_H

#include <string>
#include <vector>

#include "modules/common/proto/geometry.pb.h"
#include "modules/integration_test/perception/obstacle/model/location_model.h"
#include "modules/integration_test/perception/obstacle/model/obstacle_base_model.h"
#include "modules/msgs/perception/proto/obstacle.pb.h"

namespace roadstar {
namespace integration_test {

using PointENU = roadstar::common::PointENU;  // actually the z value here isn't
                                              // useful for now.

class PerceptionObstacleModel : public ObstacleBaseModel {
 public:
  using Obstacle = roadstar::perception::Obstacle;

  PerceptionObstacleModel()
      : sensor_source_(0), heading_(0), theta_(0), velocity_(0) {}
  explicit PerceptionObstacleModel(const json &value) {
    ObstacleBaseModel::FromJson(value);
    sensor_source_ = value["sensor_source"];
    heading_ = value["heading"];
    theta_ = value["theta"];
    velocity_ = value["velocity"];
    type_ = value["type"];
  }
  PerceptionObstacleModel(double x, double y, double z, double l, double h,
                          double w, double heading, double theta,
                          int sensor_source, double velocity)
      : ObstacleBaseModel(x, y, z, l, h, w),
        sensor_source_(sensor_source),
        heading_(heading),
        theta_(theta),
        velocity_(velocity) {}
  explicit PerceptionObstacleModel(const Obstacle &ob) {
    FromObstacleMsg(ob);
  }

  PerceptionObstacleModel *SetX(double x);
  PerceptionObstacleModel *SetY(double y);
  PerceptionObstacleModel *SetZ(double z);
  PerceptionObstacleModel *SetH(double h);
  PerceptionObstacleModel *SetL(double l);
  PerceptionObstacleModel *SetW(double w);
  PerceptionObstacleModel *SetR(double r);
  PerceptionObstacleModel *SetSensorSource(int sensor_source);
  PerceptionObstacleModel *SetTheta(double theta);
  PerceptionObstacleModel *SetHeading(double heading);
  PerceptionObstacleModel *SetVelocity(double velocity);
  PerceptionObstacleModel *SetType(int type);

  int GetSensorSource() const;
  double GetTheta() const;
  double GetHeading() const;
  double GetVelocity() const;
  int GetType() const;

  json ToJson() const override;
  void FromJson(const json &value) override;
  void FromObstacleMsg(const Obstacle &ob);
  PerceptionObstacleModel TransformToEgoFront(const LocationModel &model,
                                              const std::string &version);

  std::vector<PointENU> GetUtmTypePolygon() const;
  std::vector<PointENU> GetEgoFrontTypePolygon(
      const LocationModel &model, const std::string &version) const;

 private:
  int sensor_source_;
  double heading_;
  double theta_;
  double velocity_;
  int type_;
};

}  // namespace integration_test
}  // namespace roadstar

#endif
