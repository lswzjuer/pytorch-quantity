#include "modules/integration_test/perception/common/model/config_model.h"
#include "modules/common/log.h"

namespace roadstar {
namespace integration_test {

std::vector<std::string>* ConfigModel::GetLabelJsonFiles() {
  return &label_json_files_;
}

std::string ConfigModel::GetValueViaKey(const std::string& key) {
  if (configs_.find(key) != configs_.end()) {
    return configs_[key];
  }
  AERROR << "Error. There is no match value for key " << key;
  return "";
}

std::vector<std::string>* ConfigModel::GetBagsName() {
  return &bags_name_;
}

std::string ConfigModel::GetBagsNameString() {
  std::string names;
  std::size_t size = bags_name_.size();
  for (const auto& it : bags_name_) {
    names += it;
    size--;
    if (size != 0) {
      names += " ";
    }
  }
  return names;
}
void ConfigModel::SetLabelJsonFiles(
    const std::vector<std::string>& json_files) {
  label_json_files_ = json_files;
}

void ConfigModel::SetTrafficLightFiles(const std::vector<std::string>& files) {
  traffic_light_files_ = files;
}

const std::vector<std::string>& ConfigModel::GetTrafficLightFiles() const {
  return traffic_light_files_;
}

void ConfigModel::SetValueViakey(const std::string& key,
                                 const std::string& value) {
  configs_[key] = value;
}

}  // namespace integration_test
}  // namespace roadstar
