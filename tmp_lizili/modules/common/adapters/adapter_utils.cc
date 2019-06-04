#include "modules/common/adapters/adapter_utils.h"

namespace roadstar {
namespace common {
namespace adapter {

template <>
void GetEstimatedMsg(const double timestamp_sec,
                     ::roadstar::localization::Localization* loc) {
  double time_diff = timestamp_sec - loc->header().timestamp_sec();
  loc->mutable_header()->set_timestamp_sec(timestamp_sec);
  loc->set_utm_x(loc->utm_x() + loc->vel_x() * time_diff +
                 loc->acc_x() * time_diff * time_diff);
  loc->set_utm_y(loc->utm_y() + loc->vel_y() * time_diff +
                 loc->acc_y() * time_diff * time_diff);
  loc->set_utm_z(loc->utm_z() + loc->vel_z() * time_diff +
                 loc->acc_z() * time_diff * time_diff);
  loc->set_vel_x(loc->vel_x() + time_diff * loc->acc_x());
  loc->set_vel_y(loc->vel_y() + time_diff * loc->acc_y());
  loc->set_vel_z(loc->vel_z() + time_diff * loc->acc_z());
}

}  // namespace adapter
}  // namespace common
}  // namespace roadstar
