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

#ifndef MODULES_INTEGRATION_TEST_COMMON_INTEGRATION_TEST_GFLAGS_H_
#define MODULES_INTEGRATION_TEST_COMMON_INTEGRATION_TEST_GFLAGS_H_

#include "gflags/gflags.h"
namespace roadstar {
namespace integration_test {

DECLARE_string(integration_test_config_file);
DECLARE_string(integration_test_adapter_config_file);
DECLARE_string(integration_test_config_xml);

}  // namespace integration_test
}  // namespace roadstar

#endif /* MODULES_INTEGRATION_TEST_COMMON_INTEGRATION_TEST_GFLAGS_H_ */
