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

#include <unistd.h>

#include "gtest/gtest.h"
#include "modules/common/log.h"

namespace roadstar {
namespace common {

TEST(TimeTest, testtimer) {
  Timer timer;
  timer.start();
  usleep(100000);
  uint64_t elapsed_time = timer.end("TimerTest");
  EXPECT_TRUE(elapsed_time >= 99 && elapsed_time <= 101);
}

TEST(TimerWrapperTest, test) {
  TimerWrapper wrapper("TimerWrapperTest");
  usleep(200000);
}

TEST(PerfFunctionTest, test) {
  PERF_FUNCTION("FunctionTest");
  usleep(100000);
}

TEST(PerfBlockTest, test) {
  PERF_BLOCK_START();
  // do somethings.
  usleep(100000);
  PERF_BLOCK_END("BLOCK1");

  usleep(200000);
  PERF_BLOCK_END("BLOCK2");
}

}  // namespace common
}  // namespace roadstar
