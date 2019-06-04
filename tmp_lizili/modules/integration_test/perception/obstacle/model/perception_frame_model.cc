
#include "modules/integration_test/perception/obstacle/model/perception_frame_model.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include "modules/common/log.h"

namespace roadstar {
namespace integration_test {

void PerceptionFrameModel::AddObstacleOfUtmTypeModel(
    const PerceptionObstacleModel& model) {
  utm_modeles_.push_back(model);
}

void PerceptionFrameModel::AddObstacleOfEgoFrontTypeModel(
    const PerceptionObstacleModel& model) {
  ego_front_modeles_.push_back(model);
}

void PerceptionFrameModel::AddLocationModel(const LocationModel& model) {
  location_model_ = model;
}

LocationModel* PerceptionFrameModel::GetLocationModel() {
  return &location_model_;
}

const LocationModel* PerceptionFrameModel::GetLocationModel() const {
  return &location_model_;
}

json PerceptionFrameModel::ToJson() {
  json obstacles;
  size_t size = utm_modeles_.size();
  for (size_t i = 0; i < size; ++i) {
    json ob;
    ob["ego_front"] = ego_front_modeles_[i].ToJson();
    ob["utm"] = utm_modeles_[i].ToJson();
    obstacles.push_back(ob);
  }
  json out;
  out["obstacles"] = obstacles;
  out["localization"] = location_model_.ToJson();
  return out;
}

void PerceptionFrameModel::Print() {
  json value = ToJson();
  std::cout << std::setw(4) << "perception frame begin..... " << std::endl
            << value << std::endl
            << "perception frame end ...";
}

void PerceptionFrameModel::FromJson(const json& value) {
  try {
    json obstacles = value["obstacles"];
    size_t size = obstacles.size();
    for (size_t i = 0; i < size; ++i) {
      PerceptionObstacleModel ego_front_model;
      ego_front_model.FromJson(obstacles[i]["ego_front"]);
      ego_front_modeles_.push_back(ego_front_model);
      PerceptionObstacleModel utm_model;
      utm_model.FromJson(obstacles[i]["utm"]);
      utm_modeles_.push_back(utm_model);
    }
    json location = value["localization"];
    location_model_.FromJson(location);
  } catch (...) {
    AERROR << "Catch exception.FromJson Error.";
  }
}

PerceptionObstacleModel* PerceptionFrameModel::GetEgoFrontTypeModelAt(
    size_t i) {
  if (i < ego_front_modeles_.size()) {
    return &(ego_front_modeles_[i]);
  }
  AFATAL << "Fatal: " << i << " out of ego_front_modeles bound.";
  return nullptr;
}

const PerceptionObstacleModel* PerceptionFrameModel::GetEgoFrontTypeModelAt(
    size_t i) const {
  if (i < ego_front_modeles_.size()) {
    return &(ego_front_modeles_[i]);
  }
  AFATAL << "Fatal: " << i << " out of ego_front_modeles bound.";
  return nullptr;
}

PerceptionObstacleModel* PerceptionFrameModel::GetUtmTypeModelAt(size_t i) {
  if (i < utm_modeles_.size()) {
    return &(utm_modeles_[i]);
  }
  AFATAL << "Fatal: " << i << " out of utm_modeles bound.";
  return nullptr;
}

const PerceptionObstacleModel* PerceptionFrameModel::GetUtmTypeModelAt(
    size_t i) const {
  if (i < utm_modeles_.size()) {
    return &(utm_modeles_[i]);
  }
  AFATAL << "Fatal: " << i << " out of utm_modeles bound.";
  return nullptr;
}

size_t PerceptionFrameModel::Size() const {
  return utm_modeles_.size();
}

bool PerceptionFrameModel::ParseFromFile(const std::string& file) {
  try {
    std::ifstream in_file(file);
    if (!in_file.is_open()) {
      AERROR << "file open error. path is " << file;
      return false;
    }

    json value;
    in_file >> value;
    FromJson(value);
    in_file.close();
  } catch (...) {
    AERROR << "Catch exception. ParseFromFile. path is " << file;
    return false;
  }
  return true;
}

bool PerceptionFrameModel::SerializeToFile(const std::string& file) {
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

bool PerceptionFrameModel::ParseRawDataFromFile(const std::string& file) {
  return true;
}
bool PerceptionFrameModel::SerializeRawDataToFile(const std::string& file) {
  return true;
}
}  // namespace integration_test
}  // namespace roadstar
