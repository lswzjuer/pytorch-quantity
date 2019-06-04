
#include "modules/integration_test/perception/obstacle/model/label_frame_model.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include "modules/common/log.h"

namespace roadstar {
namespace integration_test {

void LabelFrameModel::AddObstacleOfVelodyneTypeModel(
    const LabeledObstacleModel& model) {
  velodyne_modeles_.push_back(model);
}
void LabelFrameModel::AddObstacleOfEgoFrontTypeModel(
    const LabeledObstacleModel& model) {
  ego_front_modeles_.push_back(model);
}

json LabelFrameModel::ToJson() const {
  json obstacles;
  size_t size = velodyne_modeles_.size();
  for (size_t i = 0; i < size; ++i) {
    json ob;
    ob["ego_front"] = ego_front_modeles_[i].ToJson();
    ob["velodyne"] = velodyne_modeles_[i].ToJson();
    obstacles.push_back(ob);
  }
  json out;
  out["obstacles"] = obstacles;
  out["time_stamp"] = time_stamp_;
  return out;
}

void LabelFrameModel::Print() {
  json value = ToJson();
  std::cout << std::setw(4) << "label frame begin..... " << std::endl
            << value << std::endl
            << "label frame end ...";
}

size_t LabelFrameModel::Size() const {
  return velodyne_modeles_.size();
}

bool LabelFrameModel::EraseObstacleAt(size_t index) {
  if (index >= ego_front_modeles_.size() || index >= velodyne_modeles_.size()) {
    return false;
  }
  ego_front_modeles_.erase(ego_front_modeles_.begin() + index);
  velodyne_modeles_.erase(velodyne_modeles_.begin() + index);
  return true;
}

void LabelFrameModel::FromJson(const json& value) {
  try {
    json obstacles = value["obstacles"];
    time_stamp_ = value["time_stamp"];
    size_t size = obstacles.size();
    for (size_t i = 0; i < size; ++i) {
      LabeledObstacleModel ego_front_model;
      ego_front_model.FromJson(obstacles[i]["ego_front"]);
      ego_front_modeles_.push_back(ego_front_model);
      LabeledObstacleModel velodyne_model;
      velodyne_model.FromJson(obstacles[i]["velodyne"]);
      velodyne_modeles_.push_back(velodyne_model);
    }
  } catch (...) {
    AERROR << "Error.From json fail.Json value is :" << std::endl
           << std::setw(4) << value;
  }
}

LabeledObstacleModel* LabelFrameModel::GetEgoFrontTypeModelAt(size_t i) {
  if (i < ego_front_modeles_.size()) {
    return &ego_front_modeles_[i];
  }
  AFATAL << "Fatal: " << i << " out of ego_front_modeles bound.";
  return nullptr;
}

LabeledObstacleModel* LabelFrameModel::GetVelodyneTypeModelAt(size_t i) {
  if (i < velodyne_modeles_.size()) {
    return &velodyne_modeles_[i];
  }
  AFATAL << "Fatal: " << i << " out of velodyne_modeles bound.";
  return nullptr;
}

const LabeledObstacleModel* LabelFrameModel::GetEgoFrontTypeModelAt(
    size_t i) const {
  if (i < ego_front_modeles_.size()) {
    return &ego_front_modeles_[i];
  }
  AFATAL << "Fatal: " << i << " out of ego_front_modeles bound.";
  return nullptr;
}

const LabeledObstacleModel* LabelFrameModel::GetVelodyneTypeModelAt(
    size_t i) const {
  if (i < velodyne_modeles_.size()) {
    return &velodyne_modeles_[i];
  }
  AFATAL << "Fatal: " << i << " out of velodyne_modeles bound.";
  return nullptr;
}

bool LabelFrameModel::ParseFromFile(const std::string& file) {
  std::ifstream in_file(file);
  if (!in_file.is_open()) {
    AERROR << "file open error. path is " << file;
    return false;
  }

  json value;
  in_file >> value;
  FromJson(value);
  in_file.close();
  return true;
}

bool LabelFrameModel::SerializeToFile(const std::string& file) const {
  std::ofstream o(file, std::ios_base::out | std::ios_base::trunc);
  if (!o.is_open()) {
    AERROR << "file open error. path is " << file;
    return false;
  }
  json value = ToJson();
  o << std::setw(4) << value << std::endl;
  o.close();
  return true;
}

bool LabelFrameModel::ParseRawDataFromFile(const std::string& file) {
  std::ifstream in_file(file);
  if (!in_file.is_open()) {
    AERROR << "file open error. path is " << file;
    return false;
  }

  json value;
  in_file >> value;
  FromRawDataJson(value);
  in_file.close();
  return true;
}

bool LabelFrameModel::SerializeRawDataToFile(const std::string& file) const {
  std::ofstream o(file, std::ios_base::out | std::ios_base::trunc);
  if (!o.is_open()) {
    AERROR << "file open error. path is " << file;
    return false;
  }
  json value = RawDataToJson();
  o << std::setw(4) << value << std::endl;
  o.close();
  return true;
}

json LabelFrameModel::RawDataToJson() const {
  json obstacles;
  size_t size = velodyne_modeles_.size();
  for (size_t i = 0; i < size; ++i) {
    json ob;
    ob["velodyne"] = velodyne_modeles_[i].ToJson();
    obstacles.push_back(ob);
  }
  json out;
  out["obstacles"] = obstacles;
  out["time_stamp"] = time_stamp_;
  return out;
}

void LabelFrameModel::FromRawDataJson(const json& value) {
  try {
    json obstacles = value["obstacles"];
    time_stamp_ = value["time_stamp"];
    size_t size = obstacles.size();
    for (size_t i = 0; i < size; ++i) {
      LabeledObstacleModel velodyne_model;
      velodyne_model.FromJson(obstacles[i]["velodyne"]);
      velodyne_modeles_.push_back(velodyne_model);
    }
  } catch (...) {
    AERROR << "Error.From json fail.Json value is :" << std::endl
           << std::setw(4) << value;
  }
}

void LabelFrameModel::SetTimeStamp(double value) {
  time_stamp_ = value;
}

double LabelFrameModel::GetTimeStamp() const {
  return time_stamp_;
}

bool LabelFrameModel::HasObstacleId(size_t id) const {
  for (size_t i = 0; i < velodyne_modeles_.size(); i++) {
    if (velodyne_modeles_[i].GetId() == id) return true;
  }
  return false;
}

int LabelFrameModel::GetIndexOfObstacleId(size_t id) const {
  for (size_t i = 0; i < velodyne_modeles_.size(); i++) {
    if (velodyne_modeles_[i].GetId() == id) return i;
  }
  return -1;
}

LabeledObstacleModel* LabelFrameModel::GetEgoFrontTypeModelByID(
    std::size_t id) {
  for (auto& it : ego_front_modeles_) {
    if (it.GetId() == id) {
      return &it;
    }
  }
  return nullptr;
}

LabeledObstacleModel* LabelFrameModel::GetVelodyneTypeModelByID(
    std::size_t id) {
  for (auto& it : velodyne_modeles_) {
    if (it.GetId() == id) {
      return &it;
    }
  }
  return nullptr;
}

}  // namespace integration_test
}  // namespace roadstar
