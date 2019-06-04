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

#include "modules/integration_test/common/integration_test_gflags.h"

namespace roadstar {
namespace integration_test {

DEFINE_string(integration_test_config_file,
              "modules/integration_test/perception/conf/integration.conf",
              "integration config");

DEFINE_string(integration_test_adapter_config_file,
              "modules/integration_test/perception/conf/adapter.conf",
              "integration adapter file");
DEFINE_string(integration_test_config_xml,
              "modules/integration_test/perception/scripts/param/bags_9000_00-03.xml",
              "a xml contains labeled datas and bags information for integration_test");
}  // namespace integration_test
}  // namespace roadstar
