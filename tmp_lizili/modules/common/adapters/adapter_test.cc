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

#include "modules/common/adapters/adapter.h"

#include <string>
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "modules/common/adapters/adapter_gflags.h"
#include "modules/common/adapters/proto/adapter_config.pb.h"
#include "modules/common/adapters/shared_adapter.h"
#include "modules/msgs/localization/proto/localization.pb.h"
#include "ros/include/std_msgs/Int32.h"

namespace roadstar {
namespace common {
namespace adapter {

using IntegerAdapter = Adapter<std_msgs::Int32>;

TEST(AdapterTest, Empty) {
  IntegerAdapter adapter("Integer", "integer_topic", 10);
  EXPECT_TRUE(adapter.Empty());
}

TEST(AdapterTest, Observe) {
  IntegerAdapter adapter("Integer", "integer_topic", 10);
  std_msgs::Int32 i;
  i.data = 173;
  adapter.OnReceive(i);

  // Before calling Observe.
  EXPECT_TRUE(adapter.Empty());

  // After calling Observe.
  adapter.Observe();
  EXPECT_FALSE(adapter.Empty());
}

TEST(AdapterTest, GetLatestObserved) {
  IntegerAdapter adapter("Integer", "integer_topic", 2);
  std_msgs::Int32 i;
  i.data = 173;
  adapter.OnReceive(i);
  adapter.OnReceive(i);

  adapter.Observe();
  EXPECT_FALSE(adapter.Empty());
  EXPECT_EQ(173, adapter.GetLatestObserved().data);

  i.data = 5;
  adapter.OnReceive(i);
  i.data = 6;
  adapter.OnReceive(i);
  i.data = 7;
  adapter.OnReceive(i);
  // Before calling Observe() again.
  EXPECT_FALSE(adapter.Empty());
  EXPECT_EQ(173, adapter.GetLatestObserved().data);
  adapter.Observe();
  // After calling Observe() again.
  EXPECT_FALSE(adapter.Empty());
  EXPECT_EQ(7, adapter.GetLatestObserved().data);
}

TEST(AdapterTest, History) {
  IntegerAdapter adapter("Integer", "integer_topic", 3);
  std_msgs::Int32 i;
  i.data = 1;
  adapter.OnReceive(i);
  i.data = 2;
  adapter.OnReceive(i);

  adapter.Observe();
  {
    // Currently the history contains [2, 1].
    std::vector<std::shared_ptr<std_msgs::Int32>> history(adapter.begin(),
                                                          adapter.end());
    EXPECT_EQ(2, history.size());
    EXPECT_EQ(2, history[0]->data);
    EXPECT_EQ(1, history[1]->data);
  }

  i.data = 0;
  adapter.OnReceive(i);
  adapter.OnReceive(i);
  adapter.OnReceive(i);
  adapter.OnReceive(i);
  adapter.OnReceive(i);
  adapter.OnReceive(i);
  adapter.OnReceive(i);
  i.data = 3;
  adapter.OnReceive(i);
  i.data = 4;
  adapter.OnReceive(i);
  i.data = 5;
  adapter.OnReceive(i);

  {
    // Although there are more messages, without calling Observe,
    // the history is still [2, 1].
    std::vector<std::shared_ptr<std_msgs::Int32>> history(adapter.begin(),
                                                          adapter.end());
    EXPECT_EQ(2, history.size());
    EXPECT_EQ(2, history[0]->data);
    EXPECT_EQ(1, history[1]->data);
  }

  adapter.Observe();
  {
    // After calling Observe(), the history starts from 5. Since we only
    // maintain 3 elements in this adapter, 1 and 2 will be thrown out.
    //
    // History should be 5, 4, 3.
    std::vector<std::shared_ptr<std_msgs::Int32>> history(adapter.begin(),
                                                          adapter.end());
    EXPECT_EQ(3, history.size());
    EXPECT_EQ(5, history[0]->data);
    EXPECT_EQ(4, history[1]->data);
    EXPECT_EQ(3, history[2]->data);
  }
}

TEST(AdapterTest, Callback) {
  IntegerAdapter adapter("Integer", "integer_topic", 3);

  // Set the callback to act as a counter of messages.
  int count = 0;
  adapter.AddCallback([&count](std_msgs::Int32 x) { count += x.data; });

  std_msgs::Int32 i;
  i.data = 11;
  adapter.OnReceive(i);
  i.data = 41;
  adapter.OnReceive(i);
  i.data = 31;
  adapter.OnReceive(i);
  EXPECT_EQ(11 + 41 + 31, count);
}

using MyLocalizationAdapter = Adapter<localization::Localization>;

TEST(AdapterTest, GetExceptedObserved) {
  MyLocalizationAdapter adapter("local", "local_topic", 3);

  double check_time =
      adapter.GetExpectedObserved(100.35).header().timestamp_sec();
  EXPECT_EQ(0, check_time);

  localization::Localization msg;
  msg.mutable_header()->set_sequence_num(17);
  msg.mutable_header()->set_timestamp_sec(100.25);
  adapter.OnReceive(msg);
  msg.mutable_header()->set_sequence_num(18);
  msg.mutable_header()->set_timestamp_sec(100.30);
  adapter.OnReceive(msg);
  msg.mutable_header()->set_sequence_num(19);
  msg.mutable_header()->set_timestamp_sec(100.45);
  adapter.OnReceive(msg);
  adapter.Observe();
  EXPECT_FALSE(adapter.Empty());
  check_time = adapter.GetExpectedObserved(100.35).header().timestamp_sec();
  EXPECT_EQ(100.30, check_time);
  check_time = adapter.GetExpectedObserved(-1).header().timestamp_sec();
  EXPECT_EQ(100.25, check_time);
}

class Foo : public SharedData {
 public:
  Foo() = default;
  explicit Foo(const int a) : a_(a) {}

 private:
  int a_{0};
};

using FooAdapter = Adapter<Foo>;

TEST(AdapterTest, CppAdapter) {
  FooAdapter adapter("foo", "foo_topic", 3);
  Foo data;
  data.set_timestamp_sec(100);
  adapter.OnReceive(data);
  data.set_timestamp_sec(101);
  adapter.OnReceive(data);
  double check_time = 0;
  check_time = adapter.GetExpectedObserved(100.85).timestamp_sec();
  EXPECT_EQ(101, check_time);
  check_time = adapter.GetExpectedObserved(0).timestamp_sec();
  EXPECT_EQ(100, check_time);
}

TEST(SharedAdapterTest, AddAndGet) {
  SharedAdapter::instance()->Add<Foo>("foo_topic", 3);
  auto shared_data_adapter = SharedAdapter::instance()->Get<Foo>();
  Foo data;
  data.set_timestamp_sec(100);
  shared_data_adapter->OnReceive(data);
  data.set_timestamp_sec(101);
  shared_data_adapter->OnReceive(data);
  double check_time = 0;
  check_time = shared_data_adapter->GetExpectedObserved(100.85).timestamp_sec();
  EXPECT_EQ(101, check_time);
  check_time = shared_data_adapter->GetExpectedObserved(0).timestamp_sec();
  EXPECT_EQ(100, check_time);
}

TEST(SharedAdapterTest, Publish) {
  SharedAdapter::instance()->Add<Foo>("foo_topic", 3);
  int foo_data = 1;
  Foo data(foo_data);
  data.set_timestamp_sec(100);
  PublishSharedData(data);
  data.set_timestamp_sec(101);
  PublishSharedData(data);

  auto shared_data_adapter = SharedAdapter::instance()->Get<Foo>();
  double check_time = 0;
  check_time = shared_data_adapter->GetExpectedObserved(100.85).timestamp_sec();
  EXPECT_EQ(101, check_time);
  check_time = shared_data_adapter->GetExpectedObserved(0).timestamp_sec();
  EXPECT_EQ(100, check_time);
}

}  // namespace adapter
}  // namespace common
}  // namespace roadstar
