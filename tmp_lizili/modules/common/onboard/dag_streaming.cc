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

#include "modules/common/onboard/dag_streaming.h"

#include <unistd.h>

#include <utility>

#include "modules/common/log.h"
#include "modules/common/util/file.h"

namespace roadstar {
namespace common {

using std::map;
using std::string;
using std::vector;

using google::protobuf::TextFormat;

DAGStreaming::DAGStreaming()
    : Thread(true, "DAGStreamingThread"), inited_(false) {}

DAGStreaming::~DAGStreaming() {}

bool DAGStreaming::Init(const string& dag_config_path) {
  if (inited_) {
    AWARN << "DAGStreaming Init twice.";
    return true;
  }

  DAGConfig dag_config;
  string content;
  if (!roadstar::common::util::GetContent(dag_config_path, &content)) {
    AERROR << "failed to load DAGConfig file: " << dag_config_path;
    return false;
  }

  if (!TextFormat::ParseFromString(content, &dag_config)) {
    AERROR << "failed to Parse DAGConfig proto: " << dag_config_path;
    return false;
  }

  if (!InitSubnodes(dag_config)) {
    AERROR << "failed to Init Subnode. file: " << dag_config_path;
    return false;
  }

  inited_ = true;
  AINFO << "DAGStreaming Init success.";
  return true;
}

void DAGStreaming::Schedule() {
  // start all subnodes.
  for (auto& pair : subnode_map_) {
    pair.second->Start();
  }

  AINFO << "DAGStreaming start to schedule...";

  for (auto& pair : subnode_map_) {
    pair.second->Join();
  }

  AINFO << "DAGStreaming schedule exit.";
}

void DAGStreaming::Stop() {
  // stop all subnodes.
  for (auto& pair : subnode_map_) {
    pair.second->Stop();
  }

  // sleep 100 ms
  usleep(100000);
  // kill thread which is blocked
  for (auto& pair : subnode_map_) {
    if (pair.second->IsAlive()) {
      AINFO << "pthread_cancel to thread " << pair.second->Tid();
      pthread_cancel(pair.second->Tid());
    }
  }

  AINFO << "DAGStreaming is stoped.";
}

bool DAGStreaming::InitSubnodes(const DAGConfig& dag_config) {
  const DAGConfig::SubnodeConfig& subnode_config = dag_config.subnode_config();

  map<SubnodeID, DAGConfig::Subnode> subnode_config_map;

  for (auto& subnode_proto : subnode_config.subnodes()) {
    std::pair<map<SubnodeID, DAGConfig::Subnode>::iterator, bool> result =
        subnode_config_map.insert(
            std::make_pair(subnode_proto.id(), subnode_proto));
    if (!result.second) {
      AERROR << "duplicate SubnodeID: " << subnode_proto.id();
      return false;
    }
  }

  // Generate Subnode instance.
  for (auto pair : subnode_config_map) {
    const DAGConfig::Subnode& subnode_config = pair.second;
    const SubnodeID subnode_id = pair.first;
    Subnode* inst = SubnodeRegisterer::GetInstanceByName(subnode_config.name());

    if (inst == NULL) {
      AERROR << "failed to get subnode instance. name: "
             << subnode_config.name();
      return false;
    }

    bool result = inst->Init(subnode_config);
    if (!result) {
      AERROR << "failed to Init subnode. name: " << inst->name();
      return false;
    }
    subnode_map_.emplace(subnode_id, std::unique_ptr<Subnode>(inst));
    subnode_name_map_[subnode_config.name()] = subnode_id;
    AINFO << "Init subnode succ. " << inst->DebugString();
  }

  AINFO << "DAGStreaming load " << subnode_map_.size() << " subnodes.";
  return true;
}

void DAGStreaming::Run() {
  Schedule();
}

void DAGStreaming::Reset() {
  AINFO << "DAGStreaming RESET.";
}

Subnode* DAGStreaming::GetSubnodeById(SubnodeID id) {
  if (subnode_map_.find(id) != subnode_map_.end()) {
    return subnode_map_[id].get();
  }
  return nullptr;
}

Subnode* DAGStreaming::GetSubnodeByName(std::string name) {
  std::map<std::string, SubnodeID>::iterator iter =
      subnode_name_map_.find(name);
  if (iter != subnode_name_map_.end()) {
    return subnode_map_[iter->second].get();
  }
  return nullptr;
}

}  // namespace common
}  // namespace roadstar
