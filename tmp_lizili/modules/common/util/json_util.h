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

#ifndef MODULES_COMMON_UTIL_JSON_UTIL_H_
#define MODULES_COMMON_UTIL_JSON_UTIL_H_

#include <string>
#include <vector>

#include "google/protobuf/message.h"
#include "third_party/json/json.hpp"

namespace roadstar {
namespace common {
namespace util {

class JsonUtil {
 public:
  /**
   * @brief Convert proto to a json string.
   * @return A json with two fields: {type:<json_type>, data:<proto_to_json>}.
   */
  static nlohmann::json ProtoToTypedJson(
      const std::string &json_type, const google::protobuf::Message &proto);

  /**
   * @brief Get a string value from the given json[key].
   * @return Whether the field exists and is a valid string.
   */
  static bool GetStringFromJson(const nlohmann::json &json,
                                const std::string &key, std::string *value);

  /**
   * @brief Get a string vector from the given json[key].
   * @return Whether the field exists and is a valid string vector.
   */
  static bool GetStringVectorFromJson(const nlohmann::json &json,
                                      const std::string &key,
                                      std::vector<std::string> *value);
};

}  // namespace util
}  // namespace common
}  // namespace roadstar

#endif  // MODULES_COMMON_UTIL_JSON_UTIL_H_
