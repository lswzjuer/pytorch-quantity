#include "modules/integration_test/common/xml_param/xml_param_reader.h"

namespace roadstar {
namespace integration_test {
namespace {
struct SubNodesWalker : public pugi::xml_tree_walker {
  /**
   * constructor.
   * */
  explicit SubNodesWalker(std::vector<std::string> *jsons_in)
      : jsons(jsons_in) {}
  std::vector<std::string> *jsons;
  bool for_each(pugi::xml_node &node) override {
    if (node.name() == pugi::string_t("nd")) {
      const std::string name = node.attribute("name").as_string();
      jsons->push_back(name);
    }
    return true;
  }
};
}  // namespace

void XMLParamReader::FillAttributes() {
  std::map<std::string, std::string> sub_nodes;
  sub_nodes["name"] = "vehicle";
  attributes_["//vehicle[@name]"] = sub_nodes;
  sub_nodes.clear();
  sub_nodes["path"] = "mid_file_path";
  attributes_["//mid_files[@path]"] = sub_nodes;
  sub_nodes.clear();
  sub_nodes["path"] = "report_path";
  sub_nodes["name"] = "report_name";
  attributes_["//report[@path and @name]"] = sub_nodes;
  sub_nodes.clear();
  sub_nodes["path"] = "traffic_light_report_path";
  sub_nodes["name"] = "traffic_light_report_name";
  attributes_["//traffic_light_report[@path and @name]"] = sub_nodes;
  sub_nodes.clear();
  sub_nodes["map_name"] = "map_name";
  sub_nodes["routing"] = "routing";
  attributes_["//map_conf[@map_name and @routing]"] = sub_nodes;
  sub_nodes.clear();
  sub_nodes["source"] = "calibration_source";
  sub_nodes["dest"] = "calibration_dest";
  attributes_["//calibration[@source and @dest]"] = sub_nodes;
  sub_nodes.clear();
  sub_nodes["value"] = "use_camera";
  attributes_["//use_camera[@value]"] = sub_nodes;
  sub_nodes.clear();
  sub_nodes["value"] = "perception";
  attributes_["//perception[@value]"] = sub_nodes;
  sub_nodes.clear();
  sub_nodes["forward"] = "forward_limit_distance";
  sub_nodes["back"] = "back_limit_distance";
  attributes_["//limit_distance[@forward and @back]"] = sub_nodes;
  sub_nodes.clear();
  sub_nodes["path"] = "labeled_transformed_path";
  sub_nodes["name"] = "labeled_transformed_name";
  attributes_["//transformed[@path and @name]"] = sub_nodes;
  sub_nodes.clear();
  sub_nodes["value"] = "drive_scene";
  attributes_["//drive_scene[@value]"] = sub_nodes;
  sub_nodes.clear();
  sub_nodes["value"] = "test_object";
  attributes_["//test_object[@value]"] = sub_nodes;
  sub_nodes.clear();
  sub_nodes["path"] = "output_path";
  attributes_["//output[@path]"] = sub_nodes;
  sub_nodes.clear();
  sub_nodes["run_mode"] = "server_run_mode";
  attributes_["//server[@run_mode]"] = sub_nodes;
  sub_nodes.clear();
  sub_nodes["run_mode"] = "powertrain_run_mode";
  attributes_["//powertrain[@run_mode]"] = sub_nodes;
  sub_nodes.clear();
  sub_nodes["run_mode"] = "hdmap_run_mode";
  attributes_["//hdmap[@run_mode]"] = sub_nodes;
  sub_nodes.clear();
  sub_nodes["run_mode"] = "planning_run_mode";
  attributes_["//planning[@run_mode]"] = sub_nodes;
  sub_nodes.clear();
  sub_nodes["run_mode"] = "control_run_mode";
  attributes_["//control[@run_mode]"] = sub_nodes;
  sub_nodes.clear();
  sub_nodes["run_mode"] = "dreamview_run_mode";
  attributes_["//dreamview[@run_mode]"] = sub_nodes;
  sub_nodes.clear();
  sub_nodes["run_mode"] = "node_run_mode";
  attributes_["//node[@run_mode]"] = sub_nodes;
  sub_nodes.clear();
  sub_nodes["utm_x"] = "init_utm_x";
  sub_nodes["utm_y"] = "init_utm_y";
  sub_nodes["yaw"] = "init_yaw";
  attributes_["//init[@utm_x and @utm_y and @yaw]"] = sub_nodes;
  sub_nodes.clear();
}

std::string XMLParamReader::GetBagsNameString() const {
  if (!configs_) {
    AERROR << "Error.Configs_ model is nullptr.";
    std::string str;
    return str;
  }
  return configs_->GetBagsNameString();
}

std::vector<std::string> XMLParamReader::GetBagsName() const {
  if (!configs_) {
    std::vector<std::string> bags;
    AERROR << "Error.Configs_ model is nullptr.";
    return bags;
  }
  return *(configs_->GetBagsName());
}

XMLParamReader::XMLParamReader(const std::string &xml_param_path) {
  configs_ = std::make_shared<ConfigModel>();
  loading_result_ = doc_.load_file(xml_param_path.c_str());
  if (!loading_result_) {
    AERROR << "Failed to load the xml " << xml_param_path
           << " map: " << loading_result_.description();
  }
  FillAttributes();
  ParseLabelParam();
  ParseTrafficLightParam();
  ParseBagParam();
  ParseOthersParam();
}

void XMLParamReader::ParseOthersParam() {
  for (auto &it : attributes_) {
    std::string title = it.first;
    for (pugi::xpath_node node : doc_.select_nodes(title.c_str())) {
      for (auto &sub_it : it.second) {
        std::string value =
            node.node().attribute(sub_it.first.c_str()).as_string();
        configs_->SetValueViakey(sub_it.second, value);
      }
    }
  }
}

bool XMLParamReader::IsSucceedToLoad() {
  return static_cast<bool>(loading_result_);
}

void XMLParamReader::ParseBagParam() {
  for (pugi::xpath_node bags_node : doc_.select_nodes("//bag[@path]")) {
    std::string bag_path = bags_node.node().attribute("path").as_string();
    configs_->SetValueViakey("bag_path", bag_path);
    std::vector<std::string> *bags_name = configs_->GetBagsName();
    SubNodesWalker walker(bags_name);
    bags_node.node().traverse(walker);
  }
}

void XMLParamReader::ParseLabelParam() {
  for (pugi::xpath_node label_jsons_node :
       doc_.select_nodes("//label[@path]")) {
    std::string label_path =
        label_jsons_node.node().attribute("path").as_string();
    configs_->SetValueViakey("label_path", label_path);
    std::vector<std::string> label_jsons;
    SubNodesWalker walker(&label_jsons);
    label_jsons_node.node().traverse(walker);
    configs_->SetLabelJsonFiles(label_jsons);
  }
}

void XMLParamReader::ParseTrafficLightParam() {
  for (pugi::xpath_node node : doc_.select_nodes("//traffic_light[@path]")) {
    std::string path = node.node().attribute("path").as_string();
    configs_->SetValueViakey("traffic_light_path", path);
    std::vector<std::string> files;
    SubNodesWalker walker(&files);
    node.node().traverse(walker);
    configs_->SetTrafficLightFiles(files);
  }
}

std::vector<std::string> XMLParamReader::GetLabelJsons() const {
  return *(configs_->GetLabelJsonFiles());
}

std::string XMLParamReader::GetValueViaKey(const std::string &key) const {
  std::string value = configs_->GetValueViaKey(key);
  if (value.length() != 0) {
    return value;
  }
  AERROR << "no match value of the key " << key;
  return "";
}

std::shared_ptr<ConfigModel> XMLParamReader::GetConfigs() {
  return configs_;
}

}  // namespace integration_test
}  // namespace roadstar
