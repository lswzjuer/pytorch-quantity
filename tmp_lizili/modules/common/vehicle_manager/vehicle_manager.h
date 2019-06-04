#ifndef MODULES_COMMON_VEHICLE_MANAGER_VEHICLE_MANAGER_H_
#define MODULES_COMMON_VEHICLE_MANAGER_VEHICLE_MANAGER_H_

#include <string>

#include "modules/common/macro.h"
#include "modules/common/vehicle_manager/proto/vehicle_info.pb.h"

namespace roadstar {
namespace common {

class VehicleManager {
 public:
  static double GetVehicleLength();
  static double GetVehicleWidth();
  static double GetVehicleHeight();

  static const VehicleInfo &GetVehicleInfo();

 private:
  VehicleInfo vehicle_info_;

  DECLARE_SINGLETON(VehicleManager);
};

}  // namespace common
}  // namespace roadstar

#endif
