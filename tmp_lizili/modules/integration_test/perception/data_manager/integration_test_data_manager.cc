#include "modules/integration_test/perception/data_manager/integration_test_data_manager.h"

#include <algorithm>

#include "modules/common/log.h"
#include "modules/common/util/file.h"

#include "modules/integration_test/perception/common/test_object.h"
#include "modules/integration_test/perception/obstacle/serialize/label_data_parser.h"
#include "modules/integration_test/perception/traffic_light/serialize/label_traffic_light_parser.h"

namespace roadstar {
namespace integration_test {

namespace {

void Sort(std::vector<LabelFrameModel>* datas) {
  std::sort(datas->begin(), datas->end(),
            [](const LabelFrameModel self, const LabelFrameModel other) {
              return self.GetTimeStamp() < other.GetTimeStamp();
            });
}
}  // namespace

bool IntegrationTestDataManager::is_inited_ = false;

IntegrationTestDataManager::IntegrationTestDataManager() {}

void IntegrationTestDataManager::Init(
    const IntegrationTestParam& params,
    const std::shared_ptr<ConfigModel>& configs) {
  if (is_inited_) {
    AWARN << "WARNING!!! IntegrationTestDataManager has been inited before.";
    return;
  }
  is_inited_ = true;
  configs_ = configs;
  params_ = params;
}

void IntegrationTestDataManager::AddFusionMapMsg(const FusionMap& msg) {
  FusionMap* add_map = perception_test_data_set_.add_fusion_maps();
  *add_map = msg;
  perception_obs_ += msg.obstacles_size();
}

void IntegrationTestDataManager::AddTrafficLightDetectionMsg(
    const TrafficLightDetection& msg) {
  TrafficLightDetection* add_detection =
      perception_test_data_set_.add_traffic_light_detections();
  *add_detection = msg;
}

void IntegrationTestDataManager::AddLidarDetectionResult(
    const DetectionResultProto& msg) {
  DetectionResultProto* add_lidar_detection_results =
      perception_test_data_set_.add_lidar_detection_results();
  *add_lidar_detection_results = msg;
}

void IntegrationTestDataManager::AddVelodynePacketsTimestamp(
    double time_stamp) {
  velodyne_packets_timestamps_.push_back(time_stamp);
}

const std::vector<double>& IntegrationTestDataManager::GetFrameTimeStamps() {
  return velodyne_packets_timestamps_;
}

void IntegrationTestDataManager::AddLocalizationMsg(const Localization msg) {
  Localization* add_l = perception_test_data_set_.add_localizations();
  *add_l = msg;
}

void IntegrationTestDataManager::SerializeFusionMapMsgs(
    const std::string& path) {
  AINFO << "before sort velodyne_packets_timestamps_ size = "
        << velodyne_packets_timestamps_.size()
        << " perception obs = " << perception_obs_;
  std::sort(velodyne_packets_timestamps_.begin(),
            velodyne_packets_timestamps_.end(),
            [](const double self, const double other) { return self < other; });

  AINFO << "listen completed.begin to TransformToEgoFront.";
  PackageFusionMap();
  AINFO << "After  PackageFusionMap perception data size = "
        << obstacle_frames_.size() << ". begin to write file." << std::endl
        << " the parent folder is " << path << std::endl;
  roadstar::common::util::EnsureDirectory(path);
  for (auto it = obstacle_frames_.begin(); it != obstacle_frames_.end(); ++it) {
    std::string sub_file(path + "/" + std::to_string(it->first) + ".json");
    it->second.SerializeToFile(sub_file);
  }
  AINFO << "Save data to file successfully and the output path is "
        << path.c_str() << std::endl;
}

void IntegrationTestDataManager::SerializeTrafficLightMsgs(
    const std::string& path) {
  uint32_t size = perception_test_data_set_.traffic_light_detections_size();
  AINFO << "receive traffic_light_detection msgs size = " << size;
  if (size == 0) {
    return;
  }
  roadstar::common::util::EnsureDirectory(path);
  for (uint32_t i = 0; i < size; ++i) {
    auto& it = perception_test_data_set_.traffic_light_detections(i);
    std::string sub_file(path + "/" +
                         std::to_string(it.header().timestamp_sec()) + ".json");
    PerceptionTrafficLightDetectionModel model(it);
    model.SerializeToFile(sub_file);
  }
  AINFO
      << "Save traffic_lights data to file successfully and the output path is "
      << path.c_str() << std::endl;
}

void IntegrationTestDataManager::SerializeDeepLidarEvalMsgs(
    const std::string& path) {}

void IntegrationTestDataManager::SerializePerceptionMsgs(
    const std::string& path) {
  if (!is_eval_) {
    SerializeFusionMapMsgs(path);
  } else {
    // TODO(xuguodong): implemente deep lidar eval here
    // using config_.subnodes() such as DEEP_LIDAR_SESSION_RUN
    SerializeDeepLidarEvalMsgs(path);
  }
}

std::size_t IntegrationTestDataManager::GetFrame(const FusionMap& msg,
                                                 const std::size_t& begin) {
  std::vector<double>* timestamp = velodyne_packets_timestamps_.size() > 0
                                       ? &velodyne_packets_timestamps_
                                       : &point_clouds_timestamps_;
  std::size_t i = begin;
  FusionMap* msg_ptr = const_cast<FusionMap*>(&msg);
  for (; i < timestamp->size(); ++i) {
    if ((*timestamp)[i] > msg_ptr->mutable_header()->timestamp_sec()) {
      break;
    }
  }
  if (begin == timestamp->size()) {
    return begin;
  }
  return i == 0 ? i : i - 1;
}

roadstar::localization::Localization*
IntegrationTestDataManager::GetNearestLocalization(const FusionMap& msg,
                                                   const std::size_t& begin) {
  std::size_t size = perception_test_data_set_.localizations_size();
  FusionMap* msg_ptr = const_cast<FusionMap*>(&msg);
  for (std::size_t i = begin; i < size; ++i) {
    if (msg_ptr->mutable_header()->timestamp_sec() <
        perception_test_data_set_.mutable_localizations(i)
            ->mutable_header()
            ->timestamp_sec()) {
      std::size_t index = (i == 0) ? 0 : i - 1;
      return perception_test_data_set_.mutable_localizations(index);
    }
  }
  return nullptr;
}

void IntegrationTestDataManager::PackageFusionMap() {
  int size = perception_test_data_set_.fusion_maps_size();
  AINFO << "Fusion_maps_size = " << size
        << ". Begin transform utm coords to ego_front coords...." << std::endl;
  std::size_t frame_search_begin = 0;
  std::size_t localization_search_begin = 0;
  std::string perceotion_version = configs_->GetValueViaKey("perception");
  int total_obstalces = 0;
  for (int i = 0; i < size; ++i) {
    FusionMap* fusion_map = perception_test_data_set_.mutable_fusion_maps(i);
    if (!fusion_map) {
      AERROR << "fusion_map is nullptr.";
      continue;
    }
    Localization* near_local =
        GetNearestLocalization(*fusion_map, localization_search_begin);
    if (!near_local) {
      AERROR << "near_local is nullptr. GetNearestLocolization fail.";
      continue;
    }
    ++localization_search_begin;
    std::size_t frame = GetFrame(*fusion_map, frame_search_begin);
    frame_search_begin = frame + 1;
    LocationModel location_model(*near_local);
    PerceptionFrameModel frame_model;
    for (int j = 0; j < fusion_map->obstacles_size(); ++j) {
      Obstacle* ob = fusion_map->mutable_obstacles(j);
      if (ob) {
        ++total_obstalces;
        PerceptionObstacleModel ob_model(*ob);
        frame_model.AddObstacleOfUtmTypeModel(ob_model);
        PerceptionObstacleModel ob_ego_front_model =
            ob_model.TransformToEgoFront(location_model, perceotion_version);
        frame_model.AddObstacleOfEgoFrontTypeModel(ob_ego_front_model);
      }
    }
    frame_model.AddLocationModel(location_model);
    obstacle_frames_[frame] = frame_model;
  }
  AINFO << "Tranform completed.fusion_maps_size = " << obstacle_frames_.size()
        << " total_obstalces = " << total_obstalces;
}

void IntegrationTestDataManager::SerializeLocalization(
    const std::string& path) {
  Json locations;
  int size = perception_test_data_set_.localizations_size();
  for (int i = 0; i < size; ++i) {
    double time_stamp = perception_test_data_set_.mutable_localizations(i)
                            ->mutable_header()
                            ->timestamp_sec();
    double utm_x = perception_test_data_set_.mutable_localizations(i)->utm_x();
    double utm_y = perception_test_data_set_.mutable_localizations(i)->utm_y();
    double utm_z = perception_test_data_set_.mutable_localizations(i)->utm_z();
    Json location;
    location["time_stamp"] = time_stamp;
    location["utm_x"] = utm_x;
    location["utm_z"] = utm_z;
    location["utm_y"] = utm_y;
    locations[std::to_string(i)] = location;
  }
  std::string sub_file(path + "/location.json");
  std::ofstream o(sub_file, std::ios_base::out | std::ios_base::trunc);
  o << std::setw(4) << locations << std::endl;
  o.close();
}

bool IntegrationTestDataManager::DumpLabelObstacleData() {
  label_obstacles_models_.clear();
  for (auto it : (*configs_->GetLabelJsonFiles())) {
    std::string save_path;
    std::string file = configs_->GetValueViaKey("label_path") + it;
    AINFO << "Jenkins INFO: json files " << file << std::endl;
    LabelDataParser parser(file, save_path);
    std::vector<LabelFrameModel> data = parser.ParseFramesData();
    std::copy(data.begin(), data.end(),
              std::back_inserter(label_obstacles_models_));
  }
  Sort(&label_obstacles_models_);
  AINFO << "DumpLabelData models size = " << label_obstacles_models_.size();
  return label_obstacles_models_.size() != 0;
}

bool IntegrationTestDataManager::DumpLabelTrafficLightData() {
  labeled_traffic_lights_models_.clear();
  std::string path = configs_->GetValueViaKey("traffic_light_path");
  std::string mid_file_path = configs_->GetValueViaKey("mid_file_path");
  std::string save_path = mid_file_path + "labeled_traffic_light_data";
  for (auto it : configs_->GetTrafficLightFiles()) {
    std::string parent_path = path + it;
    std::vector<std::string> files =
        roadstar::common::util::ListFiles(parent_path);
    for (auto sub_it : files) {
      if (sub_it == "." || sub_it == "..") {
        continue;
      }
      std::string file = parent_path + "/" + sub_it;
      AINFO << "Jenkins INFO: json files " << file << std::endl;
      LabelTrafficLightParser parser(file, save_path);
      parser.Save();
      const auto& data = parser.GetTrafficLights();
      std::copy(data.begin(), data.end(),
                std::back_inserter(labeled_traffic_lights_models_));
    }
  }
  std::sort(labeled_traffic_lights_models_.begin(),
            labeled_traffic_lights_models_.end(),
            [](const LabeledTrafficLightDetectionModelPtr self,
               const LabeledTrafficLightDetectionModelPtr other) {
              return self->GetTimestamp() < other->GetTimestamp();
            });
  AINFO << "DumpLabelTrafficLightData models size = "
        << labeled_traffic_lights_models_.size();
  return labeled_traffic_lights_models_.size() != 0;
}

bool IntegrationTestDataManager::PrepareData() {
  std::string test_object = configs_->GetValueViaKey("test_object");
  AINFO << "prepare data begin.test_object = " << test_object;
  int mode = std::stoi(test_object);
  if (TestObject::IsTestTrafficLightMode(mode)) {
    AINFO << "mode = " << mode << " trafficlight";
    bool success = DumpLabelTrafficLightData();
    if (!success) {
      return false;
    }
  }
  if (TestObject::IsTestObstacleMode(mode)) {
    AINFO << "mode = " << mode << " obstacle";
    bool success = DumpLabelObstacleData();
    if (!success) {
      return false;
    }
  }
  return true;
}

std::string IntegrationTestDataManager::SerializePerceptionResults() {
  std::string mid_file_path = configs_->GetValueViaKey("mid_file_path");
  std::string perception_data_path = mid_file_path + "perception_obstacle_data";
  SerializePerceptionMsgs(perception_data_path);
  SerializeLocalization(mid_file_path);
  return perception_data_path;
}

const std::vector<LabelFrameModel>&
IntegrationTestDataManager::GetLabeledObstaclesModel() const {
  return label_obstacles_models_;
}

const LabeledTrafficLightDetectionModelPtrVec
IntegrationTestDataManager::GetLabeledTrafficLightModels() const {
  return labeled_traffic_lights_models_;
}

const PerceptionTrafficLightDetectionModelPtrVec
IntegrationTestDataManager::GetPerceptionTrafficLightModels() const {
  uint32_t size = perception_test_data_set_.traffic_light_detections_size();
  AINFO << "receive traffic_light_detection msgs size = " << size;
  PerceptionTrafficLightDetectionModelPtrVec datas;
  if (size == 0) {
    return datas;
  }
  for (uint32_t i = 0; i < size; ++i) {
    auto& it = perception_test_data_set_.traffic_light_detections(i);
    PerceptionTrafficLightDetectionModelPtr model(
        new PerceptionTrafficLightDetectionModel(it));
    datas.push_back(model);
  }
  return datas;
}

}  // namespace integration_test
}  // namespace roadstar
