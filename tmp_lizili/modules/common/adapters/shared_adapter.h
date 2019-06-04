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

#ifndef MODULES_ADAPTERS_SHARED_ADAPTER_H_
#define MODULES_ADAPTERS_SHARED_ADAPTER_H_

#include <atomic>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "modules/common/adapters/adapter.h"
#include "modules/common/adapters/adapter_manager.h"
#include "modules/common/macro.h"
#include "modules/common/util/type_map.h"

namespace roadstar {
namespace common {
namespace adapter {

class SharedAdapter {
 public:
  template <typename SharedDataType>
  Adapter<SharedDataType>* Get() {
    auto it = shared_data_map_.find<SharedDataType>();
    CHECK(it != shared_data_map_.end());
    return static_cast<Adapter<SharedDataType>*>(it->second.get());
  }

  template <typename SharedDataType>
  void Add(const std::string& topic_name, int message_history_limit) {
    shared_data_map_.put<SharedDataType>(
        std::make_unique<Adapter<SharedDataType>>(topic_name, topic_name,
                                                  message_history_limit));
  }

 private:
  TypeMap<std::unique_ptr<AdapterBase>> shared_data_map_;
  DECLARE_SINGLETON(SharedAdapter);
};

template <typename SharedDataType>
void PublishSharedData(const SharedDataType& shared_data) {
  // Publish it to shared adapter
  auto shared_data_adapter = SharedAdapter::instance()->Get<SharedDataType>();
  shared_data_adapter->OnReceive(shared_data);
}

}  // namespace adapter
}  // namespace common
}  // namespace roadstar

#endif  // MODULES_ADAPTERS_SHARED_ADAPTER_H_
