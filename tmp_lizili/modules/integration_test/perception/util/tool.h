#ifndef MODULES_INTEGRATION_TEST_PERCEPTION_UTIL_TOOL_H
#define MODULES_INTEGRATION_TEST_PERCEPTION_UTIL_TOOL_H

#include <string>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "modules/common/hdmap_client/map_client.h"
#include "modules/msgs/localization/proto/localization.pb.h"

namespace roadstar {
namespace integration_test {

using Localization = roadstar::localization::Localization;

class Tool {
 public:
  bool IsUtmPointOnMap(double utm_x, double utm_y, double utm_z,
                       const Localization& locate);

  static bool IsOutOfRange(double x, double forward, double back);

  bool IsEgoFrontPtOnMap(double ego_front_x, double ego_front_y,
                         double ego_front_z, const Localization& locate,
                         const std::string& version);
  static Eigen::Vector3d Velodyne64ToEgoFront(const Eigen::Vector3d& origin,
                                              const std::string& version);
  static Eigen::Vector3d WorldToEgoFront(const Eigen::Vector3d& origin,
                                         const Localization& locate,
                                         const std::string& version);
  static Eigen::Vector3d EgoFrontToWorld(const Eigen::Vector3d& origin,
                                         const Localization& locate,
                                         const std::string& version);
  static Eigen::Vector3d Velodyne64ToWorld(const Eigen::Vector3d& origin,
                                           const Localization& locate,
                                           const std::string& version);

 private:
  roadstar::common::MapClient client_;
};

}  // namespace integration_test
}  // namespace roadstar
#endif  // MODULES_COMMON_MATH_COORDS_TRANS_H
