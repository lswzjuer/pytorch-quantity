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

#ifndef MODULES_COMMON_ONBOARD_DAG_STREAMING_H_
#define MODULES_COMMON_ONBOARD_DAG_STREAMING_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "modules/common/onboard/proto/dag_config.pb.h"

#include "modules/common/macro.h"
#include "modules/common/onboard/subnode.h"
#include "modules/common/util/thread.h"

namespace roadstar {
namespace common {

class Subnode;

typedef std::map<SubnodeID, std::unique_ptr<Subnode>> SubnodeMap;

class DAGStreaming : public Thread {
 public:
  DAGStreaming();
  virtual ~DAGStreaming();

  bool Init(const std::string &dag_config_path);

  void Stop();

  size_t NumSubnodes() const {
    return subnode_map_.size();
  }

  void Reset();

  Subnode *GetSubnodeById(SubnodeID id);

  // the subnode will be overwrite if there are different node id in the same
  // name
  Subnode *GetSubnodeByName(std::string name);

 protected:
  void Run() override;

 private:
  // start run and wait.
  void Schedule();

  bool InitSubnodes(const DAGConfig &dag_config);

  bool inited_ = false;
  // NOTE(Yangguang Li): Guarantee Sunode should be firstly called destructor.
  // Subnode depends the EventManager and SharedDataManager.
  SubnodeMap subnode_map_;

  // subnode has order, IDs define the initilization order
  std::map<std::string, SubnodeID> subnode_name_map_;

  DISALLOW_COPY_AND_ASSIGN(DAGStreaming);
};

}  // namespace common
}  // namespace roadstar

#endif  // MODULES_COMMON_ONBOARD_DAG_STREAMING_H_
