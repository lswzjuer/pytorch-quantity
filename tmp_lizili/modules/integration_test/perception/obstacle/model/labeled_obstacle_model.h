#ifndef MODULES_INTEGRATION_TEST_PERCEPTION_OBSTACLE_MODEL_LABEL_OBSTACLE_MODEL_H
#define MODULES_INTEGRATION_TEST_PERCEPTION_OBSTACLE_MODEL_LABEL_OBSTACLE_MODEL_H

#include <string>
#include <vector>

#include "modules/common/proto/geometry.pb.h"
#include "modules/integration_test/perception/obstacle/model/location_model.h"
#include "modules/integration_test/perception/obstacle/model/obstacle_base_model.h"

namespace roadstar {
namespace integration_test {

using PointENU = roadstar::common::PointENU;  // actually the z value here isn't
                                              // useful for now.

class LabeledObstacleModel : public ObstacleBaseModel {
 public:
  LabeledObstacleModel() : r_(0), id_(0), velocity_(0), heading_(0) {}
  explicit LabeledObstacleModel(const json &value) {
    ObstacleBaseModel::FromJson(value);
    r_ = value["r"];
    id_ = value["id"];
    velocity_ = value["velocity"];
    heading_ = value["heading"];
    type_ = value["type"];
  }
  LabeledObstacleModel(double x, double y, double z, double l, double h,
                       double w, double r, size_t id, double velocity,
                       double heading, int type)
      : ObstacleBaseModel(x, y, z, l, h, w),
        r_(r),
        id_(id),
        velocity_(velocity),
        heading_(heading),
        type_(type) {}
  LabeledObstacleModel *SetX(double x);
  LabeledObstacleModel *SetY(double y);
  LabeledObstacleModel *SetZ(double z);
  LabeledObstacleModel *SetH(double h);
  LabeledObstacleModel *SetL(double l);
  LabeledObstacleModel *SetW(double w);
  LabeledObstacleModel *SetR(double r);
  LabeledObstacleModel *SetId(size_t id);
  LabeledObstacleModel *SetVelocity(double velocity);
  LabeledObstacleModel *SetHeading(double heading);
  LabeledObstacleModel *SetType(int type);

  std::vector<PointENU> GetVelodyneTypePolygon() const;
  std::vector<PointENU> GetEgoFrontTypePolygon(
      const std::string &version) const;

  double GetR() const;
  size_t GetId() const;
  double GetVelocity() const;
  double GetHeading() const;
  int GetType() const;

  LabeledObstacleModel ToEgoFront(const std::string &perception_version);
  LabeledObstacleModel Velodyne64ToUtm(const std::string &perception_version,
                                       const LocationModel &location);
  json ToJson() const override;
  void FromJson(const json &value) override;

 private:
  double r_;
  size_t id_;
  double velocity_;
  double heading_;
  int type_;
};

}  // namespace integration_test
}  // namespace roadstar

#endif
