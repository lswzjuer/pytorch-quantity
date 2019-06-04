#include "modules/integration_test/perception/util/tool.h"

#include <math.h>
#include <iomanip>
#include <memory>
#include <vector>

#include "modules/common/coordinate/coord_trans.h"
#include "modules/common/coordinate/sensor_coordinate.h"
#include "modules/common/log.h"
#include "modules/common/proto/geometry.pb.h"
#include "modules/common/sensor_source.h"
#include "modules/msgs/hdmap/proto/hdmap_common.pb.h"

namespace roadstar {
namespace integration_test {

using SensorCoordinate = roadstar::common::SensorCoordinate;
using CoordTransD = roadstar::common::CoordTransD;

bool Tool::IsOutOfRange(double x, double forward, double back) {
  if (x > back && x < forward) {
    return false;
  }
  return true;
}

bool Tool::IsUtmPointOnMap(double utm_x, double utm_y, double utm_z,
                           const Localization& locate) {
  roadstar::common::PointENU pt;
  pt.set_x(utm_x);
  pt.set_y(utm_y);
  pt.set_z(utm_z);
  std::vector<roadstar::common::PointENU> pts;
  pts.emplace_back(pt);
  std::vector<roadstar::hdmap::PointInfo> pt_on_road_res;

  pt.set_x(locate.utm_x());
  pt.set_y(locate.utm_y());
  pt.set_z(locate.utm_z());
  roadstar::common::Status status =
      client_.GetPointOnRoad(pts, pt, &pt_on_road_res);
  if (!status.ok()) {
    AERROR << "Tool::IsUtmPointOnMap error.Error message = "
           << status.error_message();
    return false;
  }

  bool is_pt_on_road = false;
  if (pt_on_road_res.size() > 0) {
    is_pt_on_road = pt_on_road_res[0].map_unit().type() !=
                    roadstar::hdmap::MapUnit::MAP_UNIT_NONE;
  } else {
    AWARN << "Tool::IsPointOnMap pt_on_road_res size = 0 . is_pt_on_roadstar = "
             "false.";
  }
  return is_pt_on_road;
}

bool Tool::IsEgoFrontPtOnMap(double ego_front_x, double ego_front_y,
                             double ego_front_z, const Localization& locate,
                             const std::string& version) {
  Eigen::Vector3d origin(ego_front_x, ego_front_y, ego_front_z);
  Eigen::Vector3d ret = EgoFrontToWorld(origin, locate, version);
  return IsUtmPointOnMap(ret[0], ret[1], ret[2], locate);
}

Eigen::Vector3d Tool::Velodyne64ToEgoFront(const Eigen::Vector3d& origin,
                                           const std::string& version) {
  Eigen::Vector3d ret;
  if (version == "2") {
    CoordTransD transD = SensorCoordinate::GetCoordTrans(
        common::sensor::EgoFront, common::sensor::LidarMain);
    ret = transD.TransformCoord3d(origin);
  } else {
    assert(false);
    AERROR << "Error.No such version " << version << " defination.";
  }
  return ret;
}

Eigen::Vector3d Tool::WorldToEgoFront(const Eigen::Vector3d& origin,
                                      const Localization& locate,
                                      const std::string& version) {
  Eigen::Vector3d ret;
  if (version == "2") {
    CoordTransD transD = SensorCoordinate::GetCoordTrans(
        common::sensor::EgoFront, common::sensor::World, locate);
    ret = transD.TransformCoord3d(origin);
    // AINFO << std::setprecision(14)
    // << " WorldToEgoFront version 2 ret[0] = " << ret[0]
    // << " ret[1] = " << ret[1] << "  ret[2] = " << ret[2];
  } else {
    assert(false);
    AERROR << "Error.No such version " << version << " defination.";
  }
  return ret;
}

Eigen::Vector3d Tool::EgoFrontToWorld(const Eigen::Vector3d& origin,
                                      const Localization& locate,
                                      const std::string& version) {
  Eigen::Vector3d ret;
  if (version == "2") {
    CoordTransD transD = SensorCoordinate::GetCoordTrans(
        common::sensor::World, common::sensor::EgoFront, locate);
    ret = transD.TransformCoord3d(origin);
    // AINFO << std::setprecision(14)
    // << " EgoFrontToWorld version 2 ret[0] = " << ret[0]
    // << " ret[1] = " << ret[1] << "  ret[2] = " << ret[2];
  } else {
    assert(false);
    AERROR << "Error.No such version " << version << " defination.";
  }
  return ret;
}

Eigen::Vector3d Tool::Velodyne64ToWorld(const Eigen::Vector3d& origin,
                                        const Localization& locate,
                                        const std::string& version) {
  Eigen::Vector3d ret;
  if (version == "2") {
    CoordTransD transD = SensorCoordinate::GetCoordTrans(
        common::sensor::World, common::sensor::LidarMain, locate);
    ret = transD.TransformCoord3d(origin);
    // AINFO << std::setprecision(14)
    // << "  Velodyne64ToWorld version 2 ret[0] = " << ret[0]
    // << " ret[1] = " << ret[1] << "  ret[2] = " << ret[2];
  } else {
    assert(false);
    AERROR << "Error.No such version " << version << " defination.";
  }
  return ret;
}

}  // namespace integration_test
}  // namespace roadstar
