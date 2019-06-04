#ifndef MODULES_INTEGRATION_TEST_PERCEPTION_OBSTACLE_MODEL_LOCATION_MODEL_H
#define MODULES_INTEGRATION_TEST_PERCEPTION_OBSTACLE_MODEL_LOCATION_MODEL_H

#include "modules/msgs/localization/proto/localization.pb.h"
#include "third_party/json/json.hpp"

namespace roadstar {
namespace integration_test {

using json = nlohmann::json;

class LocationModel {
 public:
  using Localization = roadstar::localization::Localization;
  LocationModel();
  LocationModel(double x, double y, double z, double heading,
                double time_stamp);
  explicit LocationModel(const Localization& msg);
  double GetX() const;
  double GetY() const;
  double GetZ() const;
  double GetHeading() const;
  double GetTimeStamp() const;
  LocationModel* SetX(double x);
  LocationModel* SetY(double y);
  LocationModel* SetZ(double z);
  LocationModel* SetHeading(double heading);
  LocationModel* SetTimeStamp(double time_stamp);
  json ToJson();
  void FromJson(const json& out);
  void FromLocalizationMsg(const Localization& msg);
  void ToLocalizationMsg(Localization* msg) const;

 protected:
  double x_;
  double y_;
  double z_;
  double heading_;
  double time_stamp_;
};

}  // namespace integration_test
}  // namespace roadstar

#endif
