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

#include "modules/common/onboard/subnode.h"

#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"

#include "modules/common/onboard/proto/dag_config.pb.h"

#include "modules/common/log.h"
#include "modules/common/util/file.h"

namespace roadstar {
namespace common {

using google::protobuf::TextFormat;
using roadstar::common::Status;

class MySubnode : public Subnode {
 public:
  MySubnode() : Subnode() {}
  virtual ~MySubnode() {}

  Status ProcEvents() override {
    AINFO << "MySubnode proc event.";
    return Status::OK();
  }
};

TEST(SubnodeTest, testinit) {
  std::string dag_config_path =
      "resources/perception/data/dag_test/dag_camera_obstacle.config";
  std::string content;
  DAGConfig dag_config;
  ASSERT_TRUE(roadstar::common::util::GetContent(dag_config_path, &content));
  ASSERT_TRUE(TextFormat::ParseFromString(content, &dag_config));

  MySubnode my_subnode;

  EXPECT_TRUE(my_subnode.Init(dag_config.subnode_config().subnodes(0)));

  AINFO << my_subnode.DebugString();

  EXPECT_EQ(my_subnode.id(), 3);
}

}  // namespace common
}  // namespace roadstar
