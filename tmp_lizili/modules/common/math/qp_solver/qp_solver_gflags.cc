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

#include "modules/common/math/qp_solver/qp_solver_gflags.h"

// math : active set solver
DEFINE_double(default_active_set_eps_num, -1e-7,
              "qpOases wrapper error control numerator");
DEFINE_double(default_active_set_eps_den, 1e-7,
              "qpOases wrapper error control numerator");
DEFINE_double(default_active_set_eps_iter_ref, 1e-7,
              "qpOases wrapper error control numerator");
DEFINE_bool(default_enable_active_set_debug_info, false,
            "Enable print information");
