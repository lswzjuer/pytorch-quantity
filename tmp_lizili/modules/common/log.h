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

/**
 * @file
 */

#ifndef MODULES_COMMON_LOG_H_
#define MODULES_COMMON_LOG_H_

#include "modules/common/log_define.h"

#ifdef USING_G3LOG
#include "g3log/g3log.hpp"
#include "g3log/sinks/custom_sink.hpp"
#else  // USING_GLOG
#include "glog/logging.h"
#include "glog/raw_logging.h"
#endif  // USING_G3LOG

/*
 * @note The macros in glog is without GLOG_ prefix. e.g. GLOG_LOG is actually
 * LOG. But this macros is conflict with the macros in tensorflow and so an
 * modified glog version for adding a GLOG_ prefix to macros to avoid this
 * redefined issue.
 * */
#define ADEBUG GLOG_VLOG(4) << "[DEBUG] "
#define AINFO GLOG_LOG(INFO)
#define AWARN GLOG_LOG(WARNING)
#define AERROR GLOG_LOG(ERROR)
#define AFATAL GLOG_LOG(FATAL)

// LOG_IF
#define AINFO_IF(cond) GLOG_LOG_IF(INFO, cond)
#define AWARN_IF(cond) GLOG_LOG_IF(WARNING, cond)
#define AERROR_IF(cond) GLOG_LOG_IF(ERROR, cond)
#define AFATAL_IF(cond) GLOG_LOG_IF(FATAL, cond)
#define ACHECK(cond) CHECK(cond)

// LOG_EVERY_N
#define AINFO_EVERY(freq) GLOG_LOG_EVERY_N(INFO, freq)
#define AWARN_EVERY(freq) GLOG_LOG_EVERY_N(WARNING, freq)
#define AERROR_EVERY(freq) GLOG_LOG_EVERY_N(ERROR, freq)

// LOG_IF_EVERY_N
#define AINFO_IF_EVERY(condition, freq) \
  GLOG_LOG_IF_EVERY_N(INFO, condition, freq)
#define AWARN_IF_EVERY(condition, freq) \
  GLOG_LOG_IF_EVERY_N(WARNING, condition, freq)
#define AERROR_IF_EVERY(condition, freq) \
  GLOG_LOG_IF_EVERY_N(ERROR, condition, freq)

// LOG_FIRST_N
#define AINFO_FIRST(freq) LOG_FIRST_N(INFO, freq)
#define AWARN_FIRST(freq) LOG_FIRST_N(WARNING, freq)
#define AERROR_FIRST(freq) LOG_FIRST_N(WARNING, freq)

namespace roadstar {
namespace common {

void InitLogging(const char *argv0);

enum LogLevel {
  FATAL_LEVEL = 0,
  ERROR_LEVEL = 1,
  WARNING_LEVEL = 2,
  INFO_LEVEL = 3,
  DEBUG_LEVEL = 4,
};

void SetStderrLoggingLevel(LogLevel level);
}  // namespace common
}  // namespace roadstar

#endif  // MODULES_COMMON_LOG_H_
