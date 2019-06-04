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

#ifndef MODULES_COMMON_ONBOARD_SUBNODE_H_
#define MODULES_COMMON_ONBOARD_SUBNODE_H_

#include <unistd.h>
#include <cstdio>
#include <iomanip>
#include <string>
#include <vector>

#include "modules/common/log.h"
#include "modules/common/onboard/proto/dag_config.pb.h"

#include "modules/common/macro.h"
#include "modules/common/status/status.h"
#include "modules/common/util/registerer.h"
#include "modules/common/util/thread.h"

namespace roadstar {
namespace common {
using SubnodeID = int;

// @brief Subnode virtual class, all business subnodes, including SubnodeIn and
//        SubnodeOut, are derived this one.
class Subnode : public Thread {
 public:
  Subnode() : Thread(true) {}

  virtual ~Subnode() {}

  // @brief Initial DataMgr, RosIO and EventMgr.
  //        It is same for all the subnodes in one stream;
  // @return  bool
  // @retval
  virtual bool Init(const DAGConfig::Subnode &config);

  virtual void Stop() {
    stop_ = true;
  }

  // @brief Subnode process interface, should be realized in derived class.
  // @return Status.
  virtual roadstar::common::Status ProcEvents() = 0;

  SubnodeID id() const {
    return id_;
  }

  std::string name() const {
    return name_;
  }

  std::string reserve() const {
    return reserve_;
  }

  virtual std::string DebugString() const;

 protected:
  double CheckTimestamp(double timestamp, double expected_diff_sec) {
    double diff_sec = timestamp - last_timestamp_;
    AWARN_IF(diff_sec > expected_diff_sec)
        << name_
        << " timeout in update subnode timestamp. Elapse time:" << diff_sec
        << ". Cur Time:" << std::fixed << std::setprecision(9) << timestamp;
    last_timestamp_ = timestamp;
    return diff_sec;
  }

  // @brief init the inner members ( default do nothing )
  // @return true/false
  virtual bool InitInternal() {
    // do nothing.
    return true;
  }

  // @brief inner run
  void Run() override;

  // following variable can be accessed by Derived Class.
  SubnodeID id_ = 0;
  std::string name_;
  std::string reserve_;
  std::vector<std::string> process_;
  DAGConfig::SubnodeType type_ = DAGConfig::SUBNODE_NORMAL;

  double last_timestamp_ = 0;

  volatile bool stop_ = false;

 private:
  bool inited_ = false;
  DISALLOW_COPY_AND_ASSIGN(Subnode);
};

REGISTER_REGISTERER(Subnode);

#define REGISTER_SUBNODE(name) REGISTER_CLASS(Subnode, name)

// common subnode, subscribe one event, and publish one evnet,
// we implement the sub event and pub event details, you can only to
// implement the handle_event().
class CommonSubnode : public Subnode {
 public:
  CommonSubnode() : Subnode() {}
  virtual ~CommonSubnode() {}

  virtual roadstar::common::Status ProcEvents();

 private:
  DISALLOW_COPY_AND_ASSIGN(CommonSubnode);
};

// Just a sample, showing how subnode works.
// class SubnodeSample : public Subnode {
// public:
//     virtual roadstar::common::Status proc_events() {
//         // SubnodeNormal
//         event_mgr_->sub(EVENT_TYPE_A, event_a);
//         data_mgr_->get_data(data)
//         do something.
//         data_mgr_->set_data(data)
//         event_mgr_->pub(event_b);
//
//         //SubnodeIn
//         ros_io_->sub(Topic, message_a);
//         do something.
//         data_mgr_->set_data(data)
//         event_mgr_->pub(event_c);
//
//         //SubnodeOut
//         event_mgr_->sub(EVENT_TYPE_D, event_d);
//         data_mgr_->get_data(data)
//         do something.
//         ros_io_->pub(message_e);
//
//
//         printf("Process one event.\n");
//         sleep(1);
//
//         return roadstar::common::Status::OK();
//     };
// };

}  // namespace common
}  // namespace roadstar

#endif  // MODULES_COMMON_ONBOARD_SUBNODE_H_
