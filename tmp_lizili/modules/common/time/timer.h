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

#ifndef MODULES_COMMON_TIMER_H_
#define MODULES_COMMON_TIMER_H_

#include <stdint.h>
#include <chrono>
#include <string>

#include "modules/common/macro.h"

namespace roadstar {
namespace common {

using TimePoint = std::chrono::system_clock::time_point;

class Timer {
 public:
  Timer() = default;

  // no-thread safe.
  void start();

  // return the elapsed time,
  // also output msg and time in glog.
  // automatically start a new timer.
  // no-thread safe.
  uint64_t end(const std::string &msg, double duration_in_ms = -1);

  uint64_t all(const std::string &msg, double duration_in_ms = -1);

 private:
  // in ms.
  TimePoint start_time_;
  TimePoint last_time_;
  TimePoint end_time_;

  DISALLOW_COPY_AND_ASSIGN(Timer);
};

class TimerWrapper {
 public:
  explicit TimerWrapper(const std::string &msg, double duration_in_ms = -1)
      : msg_(msg), duration_in_ms_(duration_in_ms) {
    timer_.start();
  }

  ~TimerWrapper() {
    timer_.end(msg_, duration_in_ms_);
  }

 private:
  Timer timer_;
  std::string msg_;
  double duration_in_ms_;

  DISALLOW_COPY_AND_ASSIGN(TimerWrapper);
};

}  // namespace common
}  // namespace roadstar

#define PERF_FUNCTION(function_name) \
  roadstar::common::TimerWrapper _timer_wrapper_(function_name)

#define PERF_FUNCTION_EXPECT(function_name, duration_in_ms) \
  roadstar::common::TimerWrapper _timer_wrapper_(function_name, duration_in_ms)

#define PERF_BLOCK_START()         \
  roadstar::common::Timer _timer_; \
  _timer_.start()

#define PERF_BLOCK_END(msg) _timer_.end(msg)

#define PERF_BLOCK_ALL(msg) _timer_.all(msg)

#define PERF_BLOCK_END_EXPECT(msg, duration_in_ms) \
  _timer_.end(msg, duration_in_ms)

#define PERF_BLOCK_ALL_EXPECT(msg, duration_in_ms) \
  _timer_.all(msg, duration_in_ms)

#endif  // MODULES_COMMON_TIMER_H_
