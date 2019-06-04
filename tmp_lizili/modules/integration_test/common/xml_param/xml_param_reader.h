#ifndef MODULES_INTEGRATION_TEST_PERCEPTION_XML_PARAM_XML_PARAM_READER_H
#define MODULES_INTEGRATION_TEST_PERCEPTION_XML_PARAM_XML_PARAM_READER_H

#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include "modules/common/log.h"
#include "modules/integration_test/perception/common/model/config_model.h"
#include "pugixml.hpp"

namespace roadstar {
namespace integration_test {

class XMLParamReader {
 public:
  explicit XMLParamReader(const std::string& xml_param_path);

  bool IsSucceedToLoad();

  std::vector<std::string> GetLabelJsons() const;
  std::vector<std::string> GetBagsName() const;
  std::string GetBagsNameString() const;
  std::string GetValueViaKey(const std::string& key) const;
  std::shared_ptr<ConfigModel> GetConfigs();

 private:
  void FillAttributes();
  void ParseLabelParam();
  void ParseTrafficLightParam();
  void ParseBagParam();
  void ParseOthersParam();

 private:
  std::map<std::string, std::map<std::string, std::string>> attributes_;
  std::shared_ptr<ConfigModel> configs_;
  pugi::xml_document doc_;
  pugi::xml_parse_result loading_result_;
};

}  // namespace integration_test
}  // namespace roadstar

#endif  // MODULES_INTEGRATION_TEST_PERCEPTION_XML_PARAM_XML_PARAM_READER_H
