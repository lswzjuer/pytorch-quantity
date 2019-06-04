/******************************************************************************
 * Copyright 2018 The Roadstar Authors. All Rights Reserved.
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

#include "modules/common/onboard/subnode.h"

#include <sstream>

#include "modules/common/log.h"

namespace roadstar {
namespace common {

using roadstar::common::ErrorCode;
using roadstar::common::Status;
using std::vector;
using std::string;
using std::ostringstream;

bool Subnode::Init(const DAGConfig::Subnode& subnode_config) {
  name_ = subnode_config.name();
  id_ = subnode_config.id();
  reserve_ = subnode_config.reserve();
  for (auto& process_str : subnode_config.process()) {
    process_.push_back(process_str);
  }
  if (subnode_config.has_type()) {
    type_ = subnode_config.type();
  }

  if (!InitInternal()) {
    AERROR << "failed to Init inner members.";
    return false;
  }

  inited_ = true;
  return true;
}

void Subnode::Run() {
  if (!inited_) {
    AERROR << "Subnode not inited, run failed. node: <" << id_ << ", " << name_
           << ">";
    return;
  }

  if (type_ == DAGConfig::SUBNODE_IN) {
    AINFO << "Subnode == SUBNODE_IN, EXIT THREAD. subnode:" << DebugString();
    return;
  }
}

string Subnode::DebugString() const {
  ostringstream oss;
  oss << "{id: " << id_ << ", name: " << name_ << ", reserve: " << reserve_
      << ", type:" << DAGConfig::SubnodeType_Name(type_) << "}";

  return oss.str();
}

}  // namespace common
}  // namespace roadstar
