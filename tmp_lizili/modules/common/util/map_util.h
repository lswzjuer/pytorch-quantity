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

/**
 * @file
 * @brief Some map util functions.
 */

#ifndef MODULES_COMMON_UTIL_MAP_UTIL_H_
#define MODULES_COMMON_UTIL_MAP_UTIL_H_

#include "google/protobuf/stubs/map_util.h"

/**
 * @namespace roadstar::common::util
 * @brief roadstar::common::util
 */
namespace roadstar {
namespace common {
namespace util {

// Expose some useful utils from protobuf.
// Find*()
using google::protobuf::FindOrDie;
using google::protobuf::FindOrDieNoPrint;
using google::protobuf::FindWithDefault;
using google::protobuf::FindOrNull;
using google::protobuf::FindPtrOrNull;
using google::protobuf::FindLinkedPtrOrNull;
using google::protobuf::FindLinkedPtrOrDie;
using google::protobuf::FindCopy;

// Contains*()
using google::protobuf::ContainsKey;
using google::protobuf::ContainsKeyValuePair;

// Insert*()
using google::protobuf::InsertOrUpdate;
using google::protobuf::InsertOrUpdateMany;
using google::protobuf::InsertAndDeleteExisting;
using google::protobuf::InsertIfNotPresent;
using google::protobuf::InsertOrDie;
using google::protobuf::InsertOrDieNoPrint;
using google::protobuf::InsertKeyOrDie;

// Lookup*()
using google::protobuf::LookupOrInsert;
using google::protobuf::AddTokenCounts;
using google::protobuf::LookupOrInsertNew;
using google::protobuf::LookupOrInsertNewLinkedPtr;
using google::protobuf::LookupOrInsertNewSharedPtr;

// Misc Utility Functions
using google::protobuf::UpdateReturnCopy;
using google::protobuf::InsertOrReturnExisting;
using google::protobuf::EraseKeyReturnValuePtr;
using google::protobuf::InsertKeysFromMap;
using google::protobuf::AppendKeysFromMap;
using google::protobuf::AppendValuesFromMap;

}  // namespace util
}  // namespace common
}  // namespace roadstar

#endif  // MODULES_COMMON_UTIL_MAP_UTIL_H_
