/******************************************************************************
 * Copyright 2017 The Roadstar Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

#include "modules/common/util/util.h"

#include <cmath>

namespace roadstar {
namespace common {
namespace util {

SLPoint MakeSLPoint(const double s, const double l) {
  SLPoint sl;
  sl.set_s(s);
  sl.set_l(l);
  return sl;
}

PointENU MakePointENU(const double x, const double y, const double z) {
  PointENU point_enu;
  point_enu.set_x(x);
  point_enu.set_y(y);
  point_enu.set_z(z);
  return point_enu;
}

PointENU MakePointENU(const math::Vec2d &xy) {
  PointENU point_enu;
  point_enu.set_x(xy.x());
  point_enu.set_y(xy.y());
  point_enu.set_z(0.0);
  return point_enu;
}

SpeedPoint MakeSpeedPoint(const double s, const double t, const double v,
                          const double a, const double da) {
  SpeedPoint speed_point;
  speed_point.set_s(s);
  speed_point.set_t(t);
  speed_point.set_v(v);
  speed_point.set_a(a);
  speed_point.set_da(da);
  return speed_point;
}

PathPoint MakePathPoint(const double x, const double y, const double z,
                        const double theta, const double kappa,
                        const double dkappa, const double ddkappa) {
  PathPoint path_point;
  path_point.set_x(x);
  path_point.set_y(y);
  path_point.set_z(z);
  path_point.set_theta(theta);
  path_point.set_kappa(kappa);
  path_point.set_dkappa(dkappa);
  path_point.set_ddkappa(ddkappa);
  return path_point;
}

bool GetCommandOutput(const std::string &command, std::string *result) {
  constexpr int kBufferSize = 2048;
  std::array<char, kBufferSize> buffer;
  auto cmd = command + " 2>&1";
  FILE *pipe = popen(cmd.c_str(), "r");
  if (!pipe) {
    AERROR << "Could not start command";
    return false;
  }
  while (fgets(buffer.data(), kBufferSize, pipe) != NULL) {
    if (result) {
      *result += buffer.data();
    }
  }
  auto return_code = pclose(pipe);
  if (return_code) {
    AERROR << "Could not close pipe";
    return false;
  }
  return true;
}

}  // namespace util
}  // namespace common
}  // namespace roadstar
