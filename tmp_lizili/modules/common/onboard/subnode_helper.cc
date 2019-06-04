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

#include "boost/algorithm/string.hpp"

#include "gflags/gflags.h"
#include "modules/common/log.h"
#include "modules/common/onboard/subnode_helper.h"

namespace roadstar {
namespace common {

DEFINE_bool(enable_frame_ratio_control, true, "enable frame ratio control");

using boost::algorithm::is_any_of;
using boost::algorithm::split;
using boost::algorithm::trim;
using std::unordered_map;
using std::string;
using std::vector;

bool SubnodeHelper::ParseReserveField(
    const string &reserve, unordered_map<string, string> *result_map) {
  int str_len = static_cast<int>(reserve.size());
  if (str_len == 0) {
    AERROR << "reserve is empty";
    return false;
  }
  int start = -1;
  int pos = 0;
  string key;
  do {
    if (reserve[pos] == ':') {
      if (pos - start <= 1) {
        AERROR << "Invalid reserve field: " << reserve;
        return false;
      }
      key = reserve.substr(start + 1, pos - start - 1);
      trim(key);
      start = pos;
    } else if (reserve[pos] == ';' || pos == str_len - 1) {
      if (key.empty() || pos - start <= 1) {
        AERROR << "Invalid reserve field: " << reserve;
        return false;
      }
      int len = reserve[pos] == ';' ? pos - start - 1 : pos - start;
      string value = reserve.substr(start + 1, len);
      trim(value);
      (*result_map)[key] = value;
      start = pos;
    } else {
      // just pass
    }
    ++pos;
  } while (pos < str_len);
  return true;
}

bool SubnodeHelper::ExtractParams(const string &conf_str,
                                  const vector<string> &param_names,
                                  vector<string> *param_values) {
  for (auto key : param_names) {
    string val;
    if (!ExtractParam(conf_str, key, &val)) {
      return false;
    }
    param_values->push_back(val);
  }
  return true;
}

bool SubnodeHelper::ExtractParam(const string &conf_str,
                                 const string &param_name,
                                 string *param_value) {
  vector<string> fields;
  split(fields, conf_str, is_any_of("&"));
  for (auto field : fields) {
    vector<string> param_pair;
    split(param_pair, field, is_any_of("="));
    if (param_pair.size() != 2u) {
      AERROR << "Invalid param(need key=value):" << field;
      return false;
    }

    string name = param_pair[0];
    name.erase(0, name.find_first_not_of(" "));
    name.erase(name.find_last_not_of(" ") + 1);

    string value = param_pair[1];
    value.erase(0, value.find_first_not_of(" "));
    value.erase(value.find_last_not_of(" ") + 1);
    if (name == param_name) {
      *param_value = value;
      return true;
    }
  }
  AERROR << "No Param:" << param_name << " conf:" << conf_str;
  return false;
}

bool FrameSkiper::Init(const double max_ratio) {
  min_interval_ = 1 / max_ratio;
  return true;
}

bool FrameSkiper::Skip(const double ts) {
  if (!FLAGS_enable_frame_ratio_control) {
    return false;
  }

  if (ts - ts_ > min_interval_) {
    ts_ = ts;
    return false;
  } else {
    return true;
  }
}

}  // namespace common
}  // namespace roadstar
