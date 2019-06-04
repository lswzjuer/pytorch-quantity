#ifndef MODULES_COMMON_UTIL_GPU_UTIL_H_
#define MODULES_COMMON_UTIL_GPU_UTIL_H_

#include <cstdio>
#include <cstdlib>
#include <regex>
#include <string>
#include <vector>
#include "modules/common/log.h"
#include "modules/common/util/util.h"

namespace roadstar {
namespace common {
namespace util {

void QueryGPU(std::string* result);

void GetGPURemain(const std::string& nvidia_smi_result,
                  std::vector<int>* gpu_remain);

size_t AssignValidGPU(const int& assign_memory, const int& assign_id = -1);

}  // namespace util
}  // namespace common
}  // namespace roadstar

#endif
