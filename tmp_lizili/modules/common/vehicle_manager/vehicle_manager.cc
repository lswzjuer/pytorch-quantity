#include "modules/common/vehicle_manager/vehicle_manager.h"

#include "modules/common/common_gflags.h"
#include "modules/common/util/file.h"

namespace roadstar {
namespace common {

VehicleManager::VehicleManager() {
  if (!roadstar::common::util::GetProtoFromFile(FLAGS_vehicle_config_path,
                                                &vehicle_info_)) {
    AFATAL << "Unable to get vehicle info from: " << FLAGS_vehicle_config_path;
  }
}

const VehicleInfo &VehicleManager::GetVehicleInfo() {
  return instance()->vehicle_info_;
}

double VehicleManager::GetVehicleLength() {
  return instance()->vehicle_info_.length();
}

double VehicleManager::GetVehicleWidth() {
  return instance()->vehicle_info_.width();
}

double VehicleManager::GetVehicleHeight() {
  return instance()->vehicle_info_.height();
}

}  // namespace common
}  // namespace roadstar
