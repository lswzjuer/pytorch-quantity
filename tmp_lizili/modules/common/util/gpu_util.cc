#include "modules/common/util/gpu_util.h"

namespace roadstar {
namespace common {
namespace util {

void QueryGPU(std::string* result) {
  GetCommandOutput("nvidia-smi", result);
}

void GetGPURemain(const std::string& nvidia_smi_result,
                  std::vector<int>* gpu_remain) {
  std::regex reg("(\\d+)MiB / (\\d+)MiB");
  std::sregex_token_iterator iter(nvidia_smi_result.begin(),
                                  nvidia_smi_result.end(), reg, 0);
  std::sregex_token_iterator end;

  for (; iter != end; ++iter) {
    std::smatch what;
    std::string line = (*iter).str();
    if (std::regex_search(line, what, reg)) {
      int used = std::stoi(what[1]);
      int total = std::stoi(what[2]);
      gpu_remain->push_back(total - used);
    }
  }
}

size_t AssignValidGPU(const int& assign_memory, const int& assign_id) {
  std::string result;
  QueryGPU(&result);
  // Caculate remain memory of every GPU
  std::vector<int> gpu_remain;
  GetGPURemain(result, &gpu_remain);

  // Check Input
  if ((assign_id != -1) &&
      (assign_id < 0 || assign_id > static_cast<int>(gpu_remain.size()) - 1)) {
    AFATAL << "Invalid GPU num";
  }

  // Assign GPU id
  if (gpu_remain[assign_id] >= assign_memory) {
    return assign_id;
  }

  // Scan other GPU
  for (size_t i = 0; i < gpu_remain.size(); i++) {
    if (gpu_remain[i] >= assign_memory) {
      if (assign_id != -1) {
        AINFO << "GPU " << assign_id << " is full, "
              << " switch to GPU " << i;
      }
      return i;
    }
  }
  // Otherwise
  AFATAL << "No avaliable GPU";
}

}  // namespace util
}  // namespace common
}  // namespace roadstar
