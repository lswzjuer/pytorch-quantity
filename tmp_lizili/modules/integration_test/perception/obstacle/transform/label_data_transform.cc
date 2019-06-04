#include <signal.h>
#include <unistd.h>

#include "gflags/gflags.h"
#include "modules/common/log.h"

#include "modules/common/util/file.h"
#include "third_party/json/json.hpp"
#include "velodyne_msgs/PointCloud.h"

#include "modules/integration_test/common/xml_param/xml_param_reader.h"
#include "modules/integration_test/perception/common/model/config_model.h"
#include "modules/integration_test/perception/obstacle/model/label_frame_model.h"
#include "modules/integration_test/perception/obstacle/serialize/label_data_parser.h"
#include "modules/integration_test/perception/util/ros_bag_view.h"

#define SIG_MY_DEFINE_TEST (__SIGRTMIN + 10)

namespace ri = roadstar::integration_test;

std::vector<velodyne_msgs::PointCloud> point_clouds;
const char kPointCloudTopic[] = "/roadstar/drivers/velodyne64/PointCloud";

void FillTimeStamp(std::vector<ri::LabelFrameModel> *models) {
  std::size_t index = 0;
  std::size_t size = models->size();
  AINFO << "ri::LabelFrameModel size = " << size
        << " PointCloud size = " << point_clouds.size();
  if (size != point_clouds.size()) {
    AERROR << "Error. Labeled data size != point_clouds size.";
  }
  std::sort(point_clouds.begin(), point_clouds.end(),
            [](const velodyne_msgs::PointCloud self,
               const velodyne_msgs::PointCloud other) {
              return static_cast<double>(self.header.stamp.sec) +
                         static_cast<double>(self.header.stamp.nsec) * 1e-9 <
                     static_cast<double>(other.header.stamp.sec) +
                         static_cast<double>(other.header.stamp.nsec) * 1e-9;
            });
  for (const auto &it : point_clouds) {
    double time = static_cast<double>(it.header.stamp.sec) +
                  static_cast<double>(it.header.stamp.nsec) * 1e-9;
    if (size > index) {
      (*models)[index++].SetTimeStamp(time);
    } else {
      AERROR << "Error index overflow. ri::LabelFrameModel size = " << size
             << " PointCloud index = " << index;
    }
  }
}

int SaveTransformedData(const std::vector<ri::LabelFrameModel> &labeled_data,
                        const ri::XMLParamReader &configs) {
  std::string transformed_path =
      configs.GetValueViaKey("labeled_transformed_path");
  std::string transformed_name =
      configs.GetValueViaKey("labeled_transformed_name");
  std::string file_prefix = transformed_path + "//" + transformed_name;
  roadstar::common::util::EnsureDirectory(file_prefix);
  int index = 0;
  for (const auto &it : labeled_data) {
    std::string file_path(file_prefix + "//" + std::to_string(index++) +
                          ".json");
    it.SerializeRawDataToFile(file_path);
  }
  AINFO << "labeled data size = " << labeled_data.size()
        << " save path = " << file_prefix;
  return 0;
}

int main(int argc, char **argv) {
  roadstar::common::InitLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (argc == 1) {
    AERROR << "please input the config xml path of label data file "
              "parameters"
           << std::endl;
    return 0;
  }
  std::string xml_path(argv[1]);
  AINFO << "loading xml " << xml_path;
  ri::XMLParamReader reader(xml_path);
  if (!reader.IsSucceedToLoad()) {
    AERROR << "load xml config file \"" << xml_path.c_str()
           << "\" failed. exiting now...";
    return 0;
  }

  std::shared_ptr<ri::ConfigModel> configs = reader.GetConfigs();
  std::vector<ri::LabelFrameModel> labeled_data;
  {
    unsigned int first_box_id = 0;
    for (auto it : (*configs->GetLabelJsonFiles())) {
      std::string save_path;
      std::string json = configs->GetValueViaKey("label_path") + it;
      AINFO << "json files" << json << std::endl;
      ri::LabelDataParser parser(json, save_path);
      parser.ParseAndSplitByFrame(&first_box_id);
      std::vector<ri::LabelFrameModel> data =
          parser.GetLabelObstaclesVectorStyle();
      std::copy(data.begin(), data.end(), std::back_inserter(labeled_data));
    }
  }

  {
    std::string bag_path = configs->GetValueViaKey("bag_path");
    std::vector<std::string> *bag_names = configs->GetBagsName();
    for (auto name : *bag_names) {
      ri::RosBagView view(bag_path + name);
      std::vector<velodyne_msgs::PointCloud> point_cloud =
          view.GetMessage<velodyne_msgs::PointCloud>(kPointCloudTopic);
      std::copy(point_cloud.begin(), point_cloud.end(),
                std::back_inserter(point_clouds));
    }
  }

  FillTimeStamp(&labeled_data);
  SaveTransformedData(labeled_data, reader);
  return 0;
}
