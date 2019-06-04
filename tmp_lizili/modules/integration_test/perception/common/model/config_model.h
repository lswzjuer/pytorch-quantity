#ifndef MODULES_INTEGRATION_TEST_PERCEPTION_COMMON_MODEL_CONFIG_MODEL_H
#define MODULES_INTEGRATION_TEST_PERCEPTION_COMMON_MODEL_CONFIG_MODEL_H

#include <map>
#include <string>
#include <vector>

namespace roadstar {
namespace integration_test {

class ConfigModel {
 public:
  std::vector<std::string>* GetLabelJsonFiles();
  void SetLabelJsonFiles(const std::vector<std::string>& json_files);
  void SetTrafficLightFiles(const std::vector<std::string>& files);
  std::string GetValueViaKey(const std::string& key);
  void SetValueViakey(const std::string& key, const std::string& value);

  std::vector<std::string>* GetBagsName();
  const std::vector<std::string>& GetTrafficLightFiles() const;
  std::string GetBagsNameString();

 private:
  std::vector<std::string> label_json_files_;
  std::vector<std::string> traffic_light_files_;
  std::vector<std::string> bags_name_;
  std::map<std::string, std::string> configs_;
};

}  // namespace integration_test
}  // namespace roadstar

#endif
