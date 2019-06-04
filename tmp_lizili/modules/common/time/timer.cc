/******************************************************************************
 * Copyright 2017 The roadstar Authors. All Rights Reserved.
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

#include "modules/common/time/timer.h"

#include "modules/common/log.h"

namespace roadstar {
namespace common {

using std::string;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

void Timer::start() {
  start_time_ = std::chrono::system_clock::now();
  last_time_ = std::chrono::system_clock::now();
}

uint64_t Timer::end(const string &msg, double duration_in_ms) {
  end_time_ = std::chrono::system_clock::now();
  uint64_t elapsed_time =
      duration_cast<milliseconds>(end_time_ - last_time_).count();

  ADEBUG << "TIMER " << msg << " elapsed_time: " << elapsed_time << " ms";
  if (duration_in_ms >= 0 && elapsed_time > duration_in_ms) {
    AWARN << "TIMEOUT " << msg << " elapsed_time: " << elapsed_time << " ms"
          << " expect_elapsed_time: " << duration_in_ms << " ms";
  }
  // start new timer.
  last_time_ = end_time_;
  return elapsed_time;
}

uint64_t Timer::all(const string &msg, double duration_in_ms) {
  end_time_ = std::chrono::system_clock::now();
  uint64_t elapsed_time =
      duration_cast<milliseconds>(end_time_ - start_time_).count();

  ADEBUG << "TIMER " << msg << " all_time: " << elapsed_time << " ms";
  if (duration_in_ms >= 0 && elapsed_time > duration_in_ms) {
    AWARN << "TIMEOUT " << msg << " elapsed_time: " << elapsed_time << " ms"
          << " expect_elapsed_time: " << duration_in_ms << " ms";
  }

  return elapsed_time;
}

}  // namespace common
}  // namespace roadstar
