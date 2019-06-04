#include "modules/integration_test/perception/obstacle/model/location_model.h"

namespace roadstar {
namespace integration_test {

LocationModel::LocationModel() {
  x_ = 0;
  y_ = 0;
  z_ = 0;
  heading_ = 0;
  time_stamp_ = 0;
}
LocationModel::LocationModel(double x, double y, double z, double heading,
                             double time_stamp)
    : x_(x), y_(y), z_(z), heading_(heading), time_stamp_(time_stamp) {}

LocationModel::LocationModel(const Localization& msg) {
  FromLocalizationMsg(msg);
}

double LocationModel::GetX() const {
  return x_;
}

double LocationModel::GetY() const {
  return y_;
}

double LocationModel::GetZ() const {
  return z_;
}

double LocationModel::GetHeading() const {
  return heading_;
}

double LocationModel::GetTimeStamp() const {
  return time_stamp_;
}

LocationModel* LocationModel::SetX(double x) {
  x_ = x;
  return this;
}

LocationModel* LocationModel::SetY(double y) {
  y_ = y;
  return this;
}

LocationModel* LocationModel::SetZ(double z) {
  z_ = z;
  return this;
}

LocationModel* LocationModel::SetHeading(double heading) {
  heading_ = heading;
  return this;
}

LocationModel* LocationModel::SetTimeStamp(double time_stamp) {
  time_stamp_ = time_stamp;
  return this;
}

json LocationModel::ToJson() {
  json out;
  out["x"] = x_;
  out["y"] = y_;
  out["z"] = z_;
  out["time_stamp"] = time_stamp_;
  out["heading"] = heading_;
  return out;
}

void LocationModel::FromJson(const json& out) {
  x_ = out["x"];
  y_ = out["y"];
  z_ = out["z"];
  time_stamp_ = out["time_stamp"];
  heading_ = out["heading"];
}

void LocationModel::FromLocalizationMsg(const Localization& msg) {
  x_ = msg.utm_x();
  y_ = msg.utm_y();
  z_ = msg.utm_z();
  heading_ = msg.heading();
  time_stamp_ = msg.header().timestamp_sec();
}

void LocationModel::ToLocalizationMsg(Localization* msg) const {
  msg->set_utm_x(x_);
  msg->set_utm_y(y_);
  msg->set_utm_z(z_);
  msg->set_heading(heading_);
  msg->mutable_header()->set_timestamp_sec(time_stamp_);
}

}  // namespace integration_test
}  // namespace roadstar
