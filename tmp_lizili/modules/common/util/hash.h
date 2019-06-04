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

#ifndef MODULES_COMMON_UTIL_HASH_H_
#define MODULES_COMMON_UTIL_HASH_H_

#include <cstddef>

namespace roadstar {
namespace common {
namespace util {

struct EnumHash {
  template <typename T>
  std::size_t operator()(T t) const {
    return static_cast<std::size_t>(t);
  }
};
}  // namespace util
}  // namespace common
}  // namespace roadstar

// IT SHOULD BE USED OUTSIZE NAMESPACE
#define ENABLE_ENUM_HASH(enum_name)                 \
  namespace std {                                   \
  template <>                                       \
  struct hash<enum_name> {                          \
    size_t operator()(const enum_name &ele) const { \
      return static_cast<std::size_t>(ele);         \
    }                                               \
  };                                                \
  }

#endif  // MODULES_COMMON_UTIL_HASH_H_
